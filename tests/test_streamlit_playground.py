import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import yaml
from unittest import mock

import pandas as pd
from PIL import Image
from zipfile import ZipFile
from io import BytesIO
import numpy as np

from tests.test_core_functions import minimal_params

from streamlit_playground import (
    load_examples,
    initialize_marble,
    list_marble_functions,
    execute_marble_function,
    save_marble_system,
    load_marble_system,
    export_core_to_json,
    import_core_from_json,
)


def test_load_examples_csv_and_json(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("input,target\n0.1,0.2\n0.2,0.4\n")
    with open(csv_path, "r", encoding="utf-8") as f:
        ex = load_examples(f)
    assert ex == [(0.1, 0.2), (0.2, 0.4)]

    json_path = tmp_path / "data.json"
    pd.DataFrame({"input": [1, 2], "target": [2, 4]}).to_json(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        ex = load_examples(f)
    assert ex == [(1.0, 2.0), (2.0, 4.0)]


def test_load_examples_zip(tmp_path):
    img = Image.new("RGB", (2, 2), color=(255, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    zip_path = tmp_path / "data.zip"
    with ZipFile(zip_path, "w") as zf:
        zf.writestr("inputs/img1.png", buf.getvalue())
        zf.writestr("targets/img1.png", buf.getvalue())

    with open(zip_path, "rb") as f:
        ex = load_examples(f)
    assert len(ex) == 1
    assert isinstance(ex[0][0], np.ndarray)


def test_initialize_marble(tmp_path):
    cfg = {"core": minimal_params(), "brain": {"save_dir": str(tmp_path)}}
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    m = initialize_marble(str(cfg_path))
    assert len(m.get_core().neurons) > 0


def test_execute_marble_function(tmp_path):
    cfg = {"core": minimal_params(), "brain": {"save_dir": str(tmp_path)}}
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    m = initialize_marble(str(cfg_path))
    funcs = list_marble_functions()
    assert "count_marble_synapses" in funcs
    out = execute_marble_function("count_marble_synapses", m)
    assert isinstance(out, int)


def test_save_load_and_core_roundtrip(tmp_path):
    cfg = {"core": minimal_params(), "brain": {"save_dir": str(tmp_path)}}
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    m = initialize_marble(str(cfg_path))

    save_path = tmp_path / "model.pkl"
    save_marble_system(m, str(save_path))
    assert save_path.exists()

    m2 = load_marble_system(str(save_path))
    assert len(m2.get_core().neurons) == len(m.get_core().neurons)

    js = export_core_to_json(m)
    m3 = import_core_from_json(js)
    assert len(m3.get_core().neurons) == len(m.get_core().neurons)
