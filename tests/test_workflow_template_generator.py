import runpy
import sys
from pathlib import Path
import pytest
import torch
from workflow_template_generator import generate_template, list_templates


def test_list_templates_contains_expected_entries():
    names = list_templates()
    assert "classification" in names and "preprocessing" in names


def _run_script(path: Path) -> None:
    argv = sys.argv[:]
    try:
        sys.argv = [str(path)]
        runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv = argv


def test_classification_template_cpu(tmp_path):
    out = tmp_path / "classification.py"
    generate_template("classification", out)
    _run_script(out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_classification_template_gpu(tmp_path):
    out = tmp_path / "classification_gpu.py"
    generate_template("classification", out)
    _run_script(out)
