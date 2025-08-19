from pathlib import Path

from scripts.analyze_imports import analyze_repository, collect_imports


def test_collect_imports(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text("import os\nfrom math import sqrt\n")
    imports = collect_imports(sample)
    assert ("os", 1) in imports
    assert any(name == "math" for name, _ in imports)


def test_analyze_repository_detects_file() -> None:
    data = analyze_repository(Path("."))
    assert any(entry["file"] == "scripts/analyze_imports.py" for entry in data)
