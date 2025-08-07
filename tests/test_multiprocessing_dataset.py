import pytest
import torch

from process_manager import ProcessManager, SharedDataset


def square(x: torch.Tensor) -> torch.Tensor:
    return x * x


def double(x: torch.Tensor) -> torch.Tensor:
    return x * 2


def test_multiprocessing_cpu():
    data = [torch.tensor([i], dtype=torch.float32) for i in range(8)]
    dataset = SharedDataset.from_tensors(data, device="cpu")
    mgr = ProcessManager(dataset, num_workers=2)
    results = mgr.run(square, device="cpu")
    for i, res in enumerate(results):
        assert torch.equal(res, torch.tensor([i * i], dtype=torch.float32))
        assert res.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_multiprocessing_gpu():
    data = [torch.tensor([i], dtype=torch.float32) for i in range(4)]
    dataset = SharedDataset.from_tensors(data, device="cuda")
    mgr = ProcessManager(dataset, num_workers=2)
    results = mgr.run(double, device="cuda")
    for i, res in enumerate(results):
        assert res.is_cuda
        assert res.item() == i * 2
