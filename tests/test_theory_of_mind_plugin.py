import importlib

import torch

import global_workspace
import theory_of_mind


def test_tom_prediction_and_broadcast():
    importlib.reload(theory_of_mind)
    gw = global_workspace.activate(capacity=2)
    tom = theory_of_mind.activate(hidden_size=4, num_layers=1, prediction_horizon=1)
    obs = torch.randn(3, 2)
    pred = tom.observe("alice", obs)
    assert pred.shape == (1, 2)
    assert gw.queue[-1].content["character"] == "alice"
    assert torch.allclose(gw.queue[-1].content["prediction"], pred)
