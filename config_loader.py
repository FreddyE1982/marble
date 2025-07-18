from pathlib import Path
import yaml

DEFAULT_CONFIG_FILE = Path(__file__).resolve().parent / "config.yaml"

def load_config(path: str | None = None) -> dict:
    """Load configuration from a YAML file."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_FILE
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data
