import pytest
import torch

from bit_tensor_dataset import BitTensorDataset
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from tests.test_core_functions import minimal_params
from transfer_learning import transfer_dataset_knowledge


def _make_dataset(device):
    pairs = [(float(i), float(i + 1)) for i in range(3)]
    return BitTensorDataset(pairs, device=device)


def test_transfer_dataset_between_models_cpu():
    ds = _make_dataset("cpu")
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    ds2 = transfer_dataset_knowledge(ds, nb, device="cpu")
    assert len(nb.training_history) == len(ds2)
    assert all(inp.device.type == "cpu" for inp, _ in ds2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_transfer_dataset_between_models_gpu():
    ds = _make_dataset("cuda")
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    ds2 = transfer_dataset_knowledge(ds, nb, device="cuda")
    assert len(nb.training_history) == len(ds2)
    assert all(inp.device.type == "cuda" for inp, _ in ds2)
