import yaml
from config_generator import parse_manual, generate_commented_config


def test_generate_commented_config(tmp_path):
    manual = tmp_path / "manual.txt"
    manual.write_text("""section:\n  param: Example parameter.""")
    config = {'section': {'param': 1}}
    cfg = tmp_path / "config.yaml"
    with cfg.open('w') as f:
        yaml.safe_dump(config, f)
    out = tmp_path / "out.yaml"
    generate_commented_config(str(cfg), str(manual), str(out))
    contents = out.read_text()
    assert "# Example parameter." in contents
    assert "param: 1" in contents
