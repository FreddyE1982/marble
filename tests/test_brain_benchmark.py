import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import random
from marble_brain import Brain
from marble_core import Core, DataLoader
from marble_neuronenblitz import Neuronenblitz
from marble_autograd import MarbleAutogradLayer
from tests.test_core_functions import minimal_params


def test_benchmark_step_returns_metrics():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader(), benchmark_enabled=True)
    layer = MarbleAutogradLayer(brain, learning_rate=0.01)
    brain.set_autograd_layer(layer)
    example = (0.1, 0.2)
    metrics = brain.benchmark_step(example)
    assert set(metrics.keys()) == {"marble", "autograd"}
    for m in metrics.values():
        assert "loss" in m and "time" in m
        assert isinstance(m["loss"], float)
        assert isinstance(m["time"], float)


def test_benchmark_interval_iterations():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader(), benchmark_enabled=True, benchmark_interval=2)
    layer = MarbleAutogradLayer(brain, learning_rate=0.01)
    brain.set_autograd_layer(layer)
    calls = []

    def record(example):
        calls.append(example)

    brain.benchmark_step = record
    examples = [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)]
    brain.train(examples, epochs=1)
    assert len(calls) == 1
