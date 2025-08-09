import pytest
import torch

from benchmark_graph_precompile import benchmark_precompile


devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")


@pytest.mark.parametrize("device", devices)
def test_benchmark_precompile_device_parity(device: str, monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: device == "cuda")
    result = benchmark_precompile(repeats=5)
    assert set(result) == {"no_precompile", "precompiled", "speedup"}
    assert result["no_precompile"] > 0
    assert result["precompiled"] > 0
    assert result["speedup"] >= 0

