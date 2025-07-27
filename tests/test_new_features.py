import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import tempfile
from marble_core import Core, perform_message_passing
from marble_utils import (
    export_core_to_onnx,
    export_neuron_state,
    import_neuron_state,
)
from marble_base import MetricsVisualizer, MetricsAggregator
from tests.test_core_functions import minimal_params


def test_graph_pruning():
    params = minimal_params()
    core = Core(params)
    # add isolated neuron
    core.neurons.append(core.neurons[0].__class__(len(core.neurons)))
    core.prune_unused_neurons()
    assert len(core.neurons) == params["width"] * params["height"]


def test_export_import_neuron_state(tmp_path):
    params = minimal_params()
    core = Core(params)
    perform_message_passing(core)
    path = tmp_path / "state.json"
    export_neuron_state(core, path)
    # reset and load
    core2 = Core(params)
    import_neuron_state(core2, path)
    for a, b in zip(core.neurons, core2.neurons):
        assert (a.representation == b.representation).all()


def test_onnx_export(tmp_path):
    import pytest
    pytest.importorskip("onnx")
    params = minimal_params()
    core = Core(params)
    file_path = tmp_path / "model.onnx"
    export_core_to_onnx(core, file_path)
    assert file_path.exists() and file_path.stat().st_size > 0


def test_metrics_aggregator():
    v1 = MetricsVisualizer()
    v2 = MetricsVisualizer()
    v1.update({"loss": 1.0})
    v2.update({"loss": 3.0})
    agg = MetricsAggregator()
    agg.add_source(v1)
    agg.add_source(v2)
    metrics = agg.aggregate()
    assert metrics["loss"] == 2.0

def test_message_passing_progress(monkeypatch):
    params = minimal_params()
    core = Core(params)
    updates = []

    class Dummy:
        def __init__(self, *args, **kwargs):
            pass
        def __iter__(self):
            return iter(core.neurons)
        def __next__(self):
            raise StopIteration
        def close(self):
            updates.append(True)

    monkeypatch.setattr("marble_core.tqdm", lambda *a, **k: Dummy())
    perform_message_passing(core, show_progress=True)
    assert updates
