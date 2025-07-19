import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marble_core import Core, DataLoader
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain
from meta_parameter_controller import MetaParameterController
from neuromodulatory_system import NeuromodulatorySystem
from marble_base import MetricsVisualizer
from tests.test_core_functions import minimal_params


def test_meta_loss_metric_recorded():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    mpc = MetaParameterController(history_length=2)
    mv = MetricsVisualizer()
    ns = NeuromodulatorySystem()
    brain = Brain(
        core,
        nb,
        DataLoader(),
        neuromodulatory_system=ns,
        meta_controller=mpc,
        metrics_visualizer=mv,
    )

    train_examples = [(0.1, 0.2)]
    val_examples = [(0.1, 0.2)]
    brain.train(train_examples, epochs=1, validation_examples=val_examples)

    assert mv.metrics["meta_loss_avg"]
