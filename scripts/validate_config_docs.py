import argparse
from pathlib import Path
from typing import List, Tuple
import sys

if __package__ is None or __package__ == "":
    # Allow running as a standalone script
    sys.path.append(str(Path(__file__).resolve().parent))
    from list_config_keys import list_config_keys  # type: ignore
else:  # pragma: no cover - import when used as package
    from .list_config_keys import list_config_keys


def parse_configurable_parameters(path: str | Path) -> List[str]:
    """Parse CONFIGURABLE_PARAMETERS.md and return dotted keys."""
    keys: List[str] = []
    section_stack: List[str] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            # New top-level section
            section_stack = [line[3:].strip()]
        elif line.startswith("### "):
            # Nested section under current
            if section_stack:
                section_stack.append(line[4:].strip())
            else:
                section_stack = [line[4:].strip()]
        elif line.startswith("-"):
            if not section_stack:
                continue
            param = line[1:].strip()
            if not param or param.startswith("("):
                continue
            if ":" in param:
                param = param.split(":", 1)[0].strip()
            key = ".".join(section_stack + [param])
            keys.append(key)
    return sorted(keys)


def validate_config_docs(config: str | Path, params_md: str | Path) -> Tuple[List[str], List[str]]:
    """Return missing and extra keys comparing config with documentation."""
    cfg_keys = set(list_config_keys(config))
    doc_keys = set(parse_configurable_parameters(params_md))
    missing = sorted(cfg_keys - doc_keys)
    extra = sorted(doc_keys - cfg_keys)
    return missing, extra


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate CONFIGURABLE_PARAMETERS.md against config.yaml"
    )
    parser.add_argument(
        "config", nargs="?", default="config.yaml", help="Path to YAML config"
    )
    parser.add_argument(
        "params", nargs="?", default="CONFIGURABLE_PARAMETERS.md", help="Documentation file to validate"
    )
    args = parser.parse_args()

    missing, extra = validate_config_docs(args.config, args.params)
    if missing:
        print("Missing keys:")
        for key in missing:
            print(f"  {key}")
    if extra:
        print("Extra undocumented keys:")
        for key in extra:
            print(f"  {key}")
    if missing or extra:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
