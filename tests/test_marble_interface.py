import os
import sys
import yaml
from unittest.mock import patch
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import marble_imports
import marble_brain
import marble_main
import marble_interface
from marble_base import MetricsVisualizer
from tqdm import tqdm as std_tqdm
from tests.test_core_functions import minimal_params
from marble import DataLoader

from marble_interface import (
    new_marble_system,
    configure_marble_system,
    save_marble_system,
    load_marble_system,
    infer_marble_system,
    train_marble_system,
    set_dreaming,
    set_autograd,
    load_hf_dataset,
    train_from_dataframe,
    evaluate_marble_system,
    export_core_to_json,
    import_core_from_json,
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
    tensor_out = infer_marble_system(loaded, 0.1, tensor=True)
    assert not isinstance(tensor_out, float)


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


def test_load_dataset_and_dataframe_training(tmp_path):
    marble_imports.tqdm = std_tqdm
    marble_brain.tqdm = std_tqdm
    marble_main.MetricsVisualizer = MetricsVisualizer

    dummy = [{"input": 0.1, "target": 0.2}, {"input": 0.2, "target": 0.4}]
    dl = DataLoader()
    with patch(
        "marble_interface.hf_login", return_value=None
    ) as login_mock, patch(
        "marble_interface.load_dataset", return_value=dummy
    ) as ld, patch.object(
        dl, "encode", wraps=dl.encode
    ) as enc:
        pairs = load_hf_dataset("dummy", "train", streaming=True, dataloader=dl)
    login_mock.assert_called_once()
    ld.assert_called_once_with(
        "dummy", split="train", token=None, streaming=True
    )
    decoded_pairs = [(dl.decode(i), dl.decode(t)) for i, t in pairs]
    assert decoded_pairs == [(0.1, 0.2), (0.2, 0.4)]
    assert enc.call_count == 4

    cfg = {"core": minimal_params(), "brain": {"save_dir": str(tmp_path)}}
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    m = new_marble_system(str(cfg_path))
    df = pd.DataFrame(dummy)
    train_from_dataframe(m, df, epochs=1)
    mse = evaluate_marble_system(m, decoded_pairs)
    assert isinstance(mse, float)


def test_export_and_import_core(tmp_path):
    marble_imports.tqdm = std_tqdm
    marble_brain.tqdm = std_tqdm
    marble_main.MetricsVisualizer = MetricsVisualizer

    cfg = {"core": minimal_params(), "brain": {"save_dir": str(tmp_path)}}
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    m = new_marble_system(str(cfg_path))
    train_marble_system(m, [(0.1, 0.2)], epochs=1)
    js = export_core_to_json(m)
    m2 = import_core_from_json(js)
    assert len(m2.get_core().neurons) == len(m.get_core().neurons)


def test_train_autoencoder_runs(tmp_path):
    marble_imports.tqdm = std_tqdm
    marble_brain.tqdm = std_tqdm
    marble_main.MetricsVisualizer = MetricsVisualizer

    cfg = {"core": minimal_params(), "brain": {"save_dir": str(tmp_path)}}
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    m = new_marble_system(str(cfg_path))
    loss = marble_interface.train_autoencoder(m, [0.1, 0.2], epochs=1)
    assert isinstance(loss, float)
    assert loss >= 0.0
