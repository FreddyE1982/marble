import importlib

import torch

import global_workspace
import theory_of_mind


def test_tom_prediction_and_broadcast():
    importlib.reload(theory_of_mind)
    gw = global_workspace.activate(capacity=2)
    tom = theory_of_mind.activate(
        hidden_size=4,
        num_layers=1,
        prediction_horizon=1,
        memory_slots=4,
        attention_hops=2,
        mismatch_threshold=0.5,
    )
    obs = torch.randn(3, 2)
    belief = {"goal": torch.zeros(4)}
    data = theory_of_mind.ToMInput(
        agent_id="agent1",
        char_id="alice",
        observations=obs,
        belief_state=belief,
    )
    pred = tom.observe(data)
    assert pred.shape == (1, 2)
    assert gw.queue[-1].content["character"] == "alice"
    assert gw.queue[-1].content["agent"] == "agent1"
    assert torch.allclose(gw.queue[-1].content["prediction"], pred)
