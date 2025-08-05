import datetime
import shutil
from pathlib import Path

import yaml
from config_schema import validate_config_schema


def load_config_text(path: str = "config.yaml") -> str:
    """Return YAML text from ``path``.

    Parameters
    ----------
    path: str
        Path to the YAML configuration file.
    """
    return Path(path).read_text()


def save_config_text(text: str, path: str = "config.yaml") -> str:
    """Validate and persist YAML text to ``path`` with a timestamped backup.

    The function first parses ``text`` with :mod:`yaml` and validates the
    resulting object against the configuration schema.  Upon successful
    validation the original file is copied to a backup file with the current
    timestamp appended.  The new YAML data is then written back to ``path``.

    Parameters
    ----------
    text: str
        YAML content to save.
    path: str
        Destination configuration file.

    Returns
    -------
    str
        The path to the created backup file.
    """
    data = yaml.safe_load(text) or {}
    validate_config_schema(data)

    cfg_path = Path(path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = cfg_path.with_suffix(cfg_path.suffix + f".{timestamp}.bak")
    shutil.copy(cfg_path, backup_path)

    cfg_path.write_text(yaml.safe_dump(data, sort_keys=False))
    return str(backup_path)
