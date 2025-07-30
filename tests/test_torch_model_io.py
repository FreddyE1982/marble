import torch
from torch_model_io import (
    save_state_dict,
    load_state_dict,
    save_entire_model,
    load_entire_model,
    save_safetensors,
    load_safetensors,
)


class SmallModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - not needed
        return self.fc(x)


def test_save_and_load_state_dict(tmp_path):
    model = SmallModel()
    path = tmp_path / "model_sd.pt"
    save_state_dict(model, path)
    loaded = load_state_dict(SmallModel, path)
    assert isinstance(loaded, SmallModel)


def test_save_and_load_entire_model(tmp_path):
    model = SmallModel()
    path = tmp_path / "model_entire.pt"
    save_entire_model(model, path)
    loaded = load_entire_model(path)
    assert isinstance(loaded, SmallModel)


def test_save_and_load_safetensors(tmp_path):
    model = SmallModel()
    path = tmp_path / "model.safetensors"
    save_safetensors(model, path)
    loaded = load_safetensors(SmallModel, path)
    assert isinstance(loaded, SmallModel)
