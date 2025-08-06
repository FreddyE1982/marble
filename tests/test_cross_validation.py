import torch

import torch
import yaml
import torch
import yaml
from cross_validation import cross_validate, k_fold_split


def test_k_fold_split_deterministic():
    data = list(range(10))
    folds1 = k_fold_split(data, 5, seed=42)
    folds2 = k_fold_split(data, 5, seed=42)
    assert [[list(s.indices) for s in pair] for pair in folds1] == [
        [list(s.indices) for s in pair] for pair in folds2
    ]


def _train(ds, device):
    # simple linear model fit using mean ratio
    xs = torch.stack([x for x, y in ds]).to(device)
    ys = torch.stack([y for x, y in ds]).to(device)
    w = torch.sum(xs * ys) / torch.sum(xs * xs)
    return w


def _metric(model, ds, device):
    xs = torch.stack([x for x, y in ds]).to(device)
    ys = torch.stack([y for x, y in ds]).to(device)
    preds = model * xs
    return float(torch.mean((preds - ys) ** 2))


def test_cross_validate_cpu():
    dataset = [
        (torch.tensor([float(i)]), torch.tensor([float(i * 2)])) for i in range(20)
    ]
    scores = cross_validate(
        _train, _metric, dataset, folds=5, seed=0, device=torch.device("cpu")
    )
    assert len(scores) == 5
    assert max(scores) < 1e-6


def test_cross_validate_gpu():
    if not torch.cuda.is_available():
        return
    dataset = [
        (torch.tensor([float(i)]), torch.tensor([float(i * 2)])) for i in range(20)
    ]
    scores = cross_validate(
        _train, _metric, dataset, folds=4, seed=1, device=torch.device("cuda")
    )
    assert len(scores) == 4
    assert max(scores) < 1e-6


def test_cross_validate_uses_config_defaults(tmp_path, monkeypatch):
    import config_loader

    cfg = config_loader.load_config()
    cfg["cross_validation"] = {"folds": 3, "seed": 7}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    monkeypatch.setattr(config_loader, "DEFAULT_CONFIG_FILE", cfg_path)
    dataset = [
        (torch.tensor([float(i)]), torch.tensor([float(i * 2)])) for i in range(9)
    ]
    scores1 = cross_validate(_train, _metric, dataset)
    scores2 = cross_validate(_train, _metric, dataset)
    assert len(scores1) == 3
    assert scores1 == scores2
