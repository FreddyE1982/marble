import pathlib
import re
import subprocess
from typing import List

PARAM_LINE = re.compile(r"^\s*-\s*([\w\.]+)")


def parse_parameters(md_path: pathlib.Path) -> List[str]:
    """Extract parameter names from CONFIGURABLE_PARAMETERS.md."""
    params: List[str] = []
    for line in md_path.read_text().splitlines():
        match = PARAM_LINE.match(line)
        if match:
            name = match.group(1).split(":", 1)[0].strip()
            if name:
                params.append(name)
    return params


def search_param(param: str, root: pathlib.Path) -> bool:
    """Return True if *param* is found in code under *root* (excluding markdown)."""
    # search using ripgrep
    result = subprocess.run(
        [
            "rg",
            param,
            "-l",
            "--glob",
            "!CONFIGURABLE_PARAMETERS.md",
            "--glob",
            "!*.md",
        ],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return bool(result.stdout.strip())


def find_unimplemented(
    md_path: pathlib.Path, root: pathlib.Path | None = None
) -> List[str]:
    """Return parameters present in md_path but absent from code."""
    if root is None:
        root = md_path.parent
    params = parse_parameters(md_path)
    missing = []
    for p in params:
        token = p.split(".")[-1]
        if not search_param(token, root):
            missing.append(p)
    return missing


def main() -> None:
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    md_path = repo_root / "CONFIGURABLE_PARAMETERS.md"
    missing = find_unimplemented(md_path, repo_root)
    for p in missing:
        print(p)


if __name__ == "__main__":
    main()
