import logging

from marble_brain import Brain
from marble_core import Core, DataLoader
from marble_neuronenblitz import Neuronenblitz
from tests.test_core_functions import minimal_params


def test_brain_logs_at_interval(caplog):
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader(), log_interval=1)
    data = [(0.0, 0.0)]
    with caplog.at_level(logging.INFO):
        brain.train(data, epochs=2)
    messages = [r.message for r in caplog.records if "Epoch" in r.message]
    assert any("Epoch 1" in m for m in messages)
    assert any("Epoch 2" in m for m in messages)
