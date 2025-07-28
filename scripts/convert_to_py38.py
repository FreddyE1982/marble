"""Convert source files to Python 3.8 compatible type hints."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Iterable


def convert_file(path: Path) -> None:
    source = path.read_text()
    tree = ast.parse(source)
    changed = False

    class RewriteAnnotations(ast.NodeTransformer):
        def visit_AnnAssign(self, node: ast.AnnAssign):
            ann = ast.unparse(node.annotation)
            if "|" in ann:
                node.annotation = ast.parse(ann.replace("|", ", ").replace("None", "None"), mode="eval").body
                changed = True
            return node

    RewriteAnnotations().visit(tree)
    if changed:
        path.write_text(ast.unparse(tree))


def convert(paths: Iterable[str]):
    for p in paths:
        path = Path(p)
        if path.is_file() and path.suffix == ".py":
            convert_file(path)
        elif path.is_dir():
            for file in path.rglob("*.py"):
                convert_file(file)


if __name__ == "__main__":
    convert(sys.argv[1:])
