import time
import pytest
import torch

from run_profiler import RunProfiler


devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")


@pytest.mark.parametrize("device", devices)
def test_run_profiler_records_device(device: str, tmp_path) -> None:
    profiler = RunProfiler(tmp_path / "profile.json")
    dev = torch.device(device)
    profiler.start("step", dev)
    time.sleep(0.001)
    profiler.end()
    assert profiler.records[0].device == device
