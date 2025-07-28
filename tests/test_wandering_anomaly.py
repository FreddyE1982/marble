import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
from collections import deque

import numpy as np
import pytest

from marble_core import Core, Neuron
from marble_neuronenblitz import Neuronenblitz
from tests.test_core_functions import minimal_params

class DummyMV:
    def __init__(self):
        self.last = None
    def update(self, metrics):
        self.last = metrics


def create_simple_nb(threshold=1.0):
    params = minimal_params()
    core = Core(params)
    core.neurons = [Neuron(0, value=0.0)]
    nb = Neuronenblitz(core, wander_anomaly_threshold=threshold, metrics_visualizer=DummyMV())
    return nb


def test_detect_wandering_anomaly():
    nb = create_simple_nb(threshold=1.0)
    for _ in range(10):
        nb.detect_wandering_anomaly(5)
    assert not nb.detect_wandering_anomaly(6)
    assert nb.detect_wandering_anomaly(12)
    assert nb.metrics_visualizer.last is not None

