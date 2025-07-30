import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline import Pipeline
from marble_main import MARBLE
from marble_interface import count_marble_synapses
from tests.test_core_functions import minimal_params


def test_pipeline_run(tmp_path):
    params = minimal_params()
    marble = MARBLE(params)
    pipe = Pipeline([{"func": "count_marble_synapses"}])
    results = pipe.execute(marble)
    assert len(results) == 1
    assert isinstance(results[0], int)


def test_pipeline_save_load(tmp_path):
    pipe = Pipeline([{"func": "count_marble_synapses"}])
    path = tmp_path / "p.json"
    pipe.save_json(path)
    with open(path, "r", encoding="utf-8") as f:
        loaded = Pipeline.load_json(f)
    assert loaded.steps == pipe.steps


