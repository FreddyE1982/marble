import sys
import yaml
import pipeline_cli
from pipeline import Pipeline


def _run(tmp_path, monkeypatch):
    pipe = Pipeline([])
    pipe_path = tmp_path / "pipe.json"
    pipe_path.write_text(pipe.to_json())
    cfg = {"pipeline": {"cache_dir": "cache_test", "default_step_memory_limit_mb": 123}}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.dump(cfg))
    captured = {}

    def fake_execute(self, marble, *, cache_dir=None, default_memory_limit_mb=None, **kwargs):
        captured["cache_dir"] = cache_dir
        captured["default"] = default_memory_limit_mb

    monkeypatch.setattr(Pipeline, "execute", fake_execute)
    monkeypatch.setattr(pipeline_cli, "create_marble_from_config", lambda path: object())
    monkeypatch.setattr(sys, "argv", ["pipeline_cli.py", str(pipe_path), "--config", str(cfg_path)])
    pipeline_cli.main()
    return captured


def test_pipeline_cli_uses_config(tmp_path, monkeypatch):
    out = _run(tmp_path, monkeypatch)
    assert out["cache_dir"] == "cache_test"
    assert out["default"] == 123
