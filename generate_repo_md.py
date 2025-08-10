#!/usr/bin/env python3
"""Generate a single Markdown file containing the entire repository.

This script walks the repository, concatenating the contents of all Markdown
files followed by all other files into a ``repo.md`` located at the repository
root. Existing ``repo.md`` is overwritten and excluded from the aggregation.
``.git`` directories are ignored.
"""
from __future__ import annotations

from pathlib import Path
from typing import List


def _collect_files(root: Path) -> tuple[List[Path], List[Path]]:
    md_files: List[Path] = []
    other_files: List[Path] = []
    excluded_dirs = {'.git', '.pytest_cache'}
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        if any(part in excluded_dirs for part in path.parts):
            continue
        rel = path.relative_to(root)
        if rel.name == 'repo.md':
            continue
        if path.suffix.lower() == '.md':
            md_files.append(rel)
        else:
            other_files.append(rel)
    md_files.sort()
    other_files.sort()
    return md_files, other_files


def build_repo_markdown(root: Path) -> Path:
    """Build ``repo.md`` for ``root`` and return its path."""
    output = root / 'repo.md'
    if output.exists():
        output.unlink()
    md_files, other_files = _collect_files(root)
    with output.open('w', encoding='utf-8') as out:
        for rel in md_files:
            out.write(f"\n# {rel}\n\n")
            out.write((root / rel).read_text(encoding='utf-8', errors='replace'))
            out.write('\n')
        for rel in other_files:
            out.write(f"\n# {rel}\n\n```{rel.suffix.lstrip('.')}\n")
            out.write((root / rel).read_text(encoding='utf-8', errors='replace'))
            out.write("\n```\n")
    return output


def main() -> None:
    root = Path(__file__).resolve().parent
    build_repo_markdown(root)


if __name__ == '__main__':
    main()
