import ast
import importlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]


def collect_imports(path: Path) -> List[Tuple[str, int]]:
    """Return list of (module, line) for imports in *path*."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    imports: List[Tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append((node.module, node.lineno))
    return imports


def check_module(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def analyze_repository(root: Path) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for file in root.rglob("*.py"):
        if any(part.startswith(".") for part in file.parts):
            continue
        imports = collect_imports(file)
        if not imports:
            continue
        entries = []
        for module, lineno in imports:
            ok = check_module(module.split(".")[0])
            entries.append({"module": module, "line": lineno, "ok": ok})
        results.append({"file": str(file.relative_to(root)), "imports": entries})
    return results


def main() -> None:
    data = analyze_repository(ROOT)
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
