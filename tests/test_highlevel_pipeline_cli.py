import subprocess
import sys
from highlevel_pipeline import HighLevelPipeline


def test_cli_checkpoint_and_resume(tmp_path):
    pipe = HighLevelPipeline(dataset_version="v42")
    json_path = tmp_path / "pipe.json"
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(pipe.to_json())
    ckpt_path = tmp_path / "pipe.pkl"

    subprocess.run(
        [
            sys.executable,
            "highlevel_pipeline_cli.py",
            "checkpoint",
            str(json_path),
            str(ckpt_path),
            "--config",
            "config.yaml",
            "--dataset-version",
            "v42",
            "--device",
            "cpu",
        ],
        check=True,
    )

    assert ckpt_path.exists()

    subprocess.run(
        [
            sys.executable,
            "highlevel_pipeline_cli.py",
            "resume",
            str(ckpt_path),
            "--config",
            "config.yaml",
            "--device",
            "cpu",
        ],
        check=True,
    )

    loaded = HighLevelPipeline.load_checkpoint(str(ckpt_path))
    assert loaded.dataset_version == "v42"
