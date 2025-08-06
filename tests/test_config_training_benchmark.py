import torch
from pathlib import Path

import pytest

from benchmark_config_training import benchmark_training_configs


@pytest.mark.parametrize(
    "device",
    [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else []),
)
def test_benchmark_training_configs(tmp_path, device):
    if hasattr(torch, "set_default_device"):
        torch.set_default_device(device)
    dataset = [(0.0, 0.0), (1.0, 1.0)]
    cfg1 = tmp_path / "config1.yaml"
    cfg2 = tmp_path / "config2.yaml"
    base = "brain:\n  manual_seed: 0\n"
    cfg1.write_text(base + "neuronenblitz:\n  learning_rate: 0.01\n")
    cfg2.write_text(base + "neuronenblitz:\n  learning_rate: 0.02\n")
    results, best_path = benchmark_training_configs(dataset, [str(cfg1), str(cfg2)])
    assert str(cfg1) in results and str(cfg2) in results
    assert (tmp_path / "best_config.yaml").exists()
    assert Path(best_path).name == "best_config.yaml"
    assert not (cfg1.exists() and cfg2.exists())
