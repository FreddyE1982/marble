import sys
import pytest
import torch

import highlevel_pipeline_cli
from highlevel_pipeline import HighLevelPipeline


def _run(monkeypatch, *args):
    monkeypatch.setattr(
        highlevel_pipeline_cli,
        "create_marble_from_config",
        lambda path: object(),
    )
    monkeypatch.setattr(sys, "argv", ["highlevel_pipeline_cli.py", *args])
    highlevel_pipeline_cli.main()


def test_cli_checkpoint_and_resume(tmp_path, monkeypatch):
    pipe = HighLevelPipeline(dataset_version="v42")
    json_path = tmp_path / "pipe.json"
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(pipe.to_json())
    ckpt_path = tmp_path / "pipe.pkl"

    _run(
        monkeypatch,
        "checkpoint",
        str(json_path),
        str(ckpt_path),
        "--config",
        "dummy.yaml",
        "--dataset-version",
        "v42",
        "--device",
        "cpu",
    )

    assert ckpt_path.exists()

    _run(
        monkeypatch,
        "resume",
        str(ckpt_path),
        "--config",
        "dummy.yaml",
        "--device",
        "cpu",
    )

    loaded = HighLevelPipeline.load_checkpoint(str(ckpt_path))
    assert loaded.dataset_version == "v42"


def test_cli_checkpoint_and_resume_gpu(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    pipe = HighLevelPipeline(dataset_version="v99")
    json_path = tmp_path / "pipe.json"
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(pipe.to_json())
    ckpt_path = tmp_path / "pipe.pkl"

    _run(
        monkeypatch,
        "checkpoint",
        str(json_path),
        str(ckpt_path),
        "--config",
        "dummy.yaml",
        "--dataset-version",
        "v99",
        "--device",
        "gpu",
    )

    _run(
        monkeypatch,
        "resume",
        str(ckpt_path),
        "--config",
        "dummy.yaml",
        "--device",
        "gpu",
    )

    loaded = HighLevelPipeline.load_checkpoint(str(ckpt_path))
    assert loaded.dataset_version == "v99"

