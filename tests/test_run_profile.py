import os, sys, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pytest
from pipeline import Pipeline


def step_a(*, device: str = "cpu"):
    return torch.tensor(1, device=device)


def step_b(*, device: str = "cpu"):
    return torch.tensor(2, device=device)


def test_run_profile_records_order(tmp_path):
    profile = tmp_path / "profile.json"
    pipe = Pipeline([
        {"module": __name__, "func": "step_a", "name": "first"},
        {"module": __name__, "func": "step_b", "name": "second"},
    ])
    pipe.execute(run_profile_path=profile)
    data = json.loads(profile.read_text())
    assert [r["step"] for r in data] == ["first", "second"]
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert all(r["device"] == expected for r in data)
