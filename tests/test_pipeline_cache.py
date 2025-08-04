import sys

import torch

from pipeline import Pipeline


def produce_number(device="cpu"):
    return torch.tensor(7, device=device)


def test_pipeline_caches_step_results(tmp_path, monkeypatch):
    pipe = Pipeline()
    pipe.add_step("produce_number", module="tests.test_pipeline_cache")
    result1 = pipe.execute(cache_dir=tmp_path)
    assert result1[0].item() == 7

    # replace function to ensure cached value is used
    def fail(device="cpu"):
        raise AssertionError("step re-executed despite cache")

    monkeypatch.setattr(sys.modules[__name__], "produce_number", fail)
    result2 = pipe.execute(cache_dir=tmp_path)
    assert result2[0].item() == 7
    files = list(tmp_path.iterdir())
    assert files, "cache file not created"
    cached = torch.load(
        files[0],
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    assert cached.item() == 7
