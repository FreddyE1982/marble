import yaml
import argparse
from typing import Dict, Any


def parse_manual(manual_path: str) -> Dict[str, str]:
    """Parse yaml-manual.txt and return mapping of section.param to description."""
    param_desc: Dict[str, str] = {}
    current_section = ""
    current_key = None
    with open(manual_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            if not line.startswith("  ") and line.rstrip().endswith(":"):
                current_section = line.strip().rstrip(":")
                current_key = None
                continue
            if line.startswith("  ") and not line.startswith("    "):
                # new parameter
                key, desc = line.strip().split(":", 1)
                current_key = f"{current_section}.{key}"
                param_desc[current_key] = desc.strip()
            elif line.startswith("    ") and current_key:
                param_desc[current_key] += " " + line.strip()
    return param_desc


def _generate_lines(data: Any, descs: Dict[str, str], section: str = "", indent: int = 0) -> list[str]:
    lines = []
    prefix = " " * indent
    if isinstance(data, dict):
        for key, val in data.items():
            full = f"{section}.{key}" if section else key
            desc = descs.get(full)
            if desc:
                lines.append(f"{prefix}# {desc}")
            if isinstance(val, dict):
                lines.append(f"{prefix}{key}:")
                lines.extend(_generate_lines(val, descs, full, indent + 2))
            else:
                dumped = yaml.safe_dump(val, default_flow_style=True).strip()
                lines.append(f"{prefix}{key}: {dumped}")
    return lines


def generate_commented_config(config_path: str, manual_path: str, output_path: str) -> None:
    """Generate a YAML config with inline comments from manual descriptions."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    descs = parse_manual(manual_path)
    lines = []
    for section, content in config.items():
        lines.append(f"{section}:")
        lines.extend(_generate_lines(content, descs, section, 2))
        lines.append("")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate commented config")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--manual", default="yaml-manual.txt")
    parser.add_argument("--output", default="sample_config_with_comments.yaml")
    args = parser.parse_args()
    generate_commented_config(args.config, args.manual, args.output)
