import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline import Pipeline
from marble_main import MARBLE
from marble_interface import count_marble_synapses
from tests.test_core_functions import minimal_params
import global_workspace
from marble_base import MetricsVisualizer


def test_pipeline_run(tmp_path):
    params = minimal_params()
    marble = MARBLE(params, dataloader_params={})
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


def test_pipeline_freeze_and_defrost(tmp_path):
    params = minimal_params()
    marble = MARBLE(params, dataloader_params={})
    pipe = Pipeline([
        {"func": "count_marble_synapses"},
        {"func": "count_marble_synapses"},
    ])
    pipe.freeze_step(1)
    results = pipe.execute(marble)
    assert len(results) == 1
    pipe.defrost_step(1)
    results = pipe.execute(marble)
    assert len(results) == 2


def test_pipeline_benchmark_and_preallocate(tmp_path):
    params = minimal_params()
    marble = MARBLE(params, dataloader_params={})
    pipe = Pipeline([{"func": "count_marble_synapses"}])
    mv = MetricsVisualizer()
    pipe.execute(
        marble,
        metrics_visualizer=mv,
        benchmark_iterations=1,
        preallocate_neurons=1,
        preallocate_synapses=1,
    )
    assert pipe.benchmarks()
    assert mv.metrics["pipeline_step"]


def test_pipeline_progress_broadcast(tmp_path):
    params = minimal_params()
    marble = MARBLE(params, dataloader_params={})
    gw = global_workspace.activate(capacity=2)
    pipe = Pipeline([{"func": "count_marble_synapses"}])
    pipe.execute(marble)
    assert gw.queue


