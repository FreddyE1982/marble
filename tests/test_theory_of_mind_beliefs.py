import importlib

import torch

import theory_of_mind


def test_memory_slot_creation_and_retrieval():
    importlib.reload(theory_of_mind)
    tom = theory_of_mind.activate(
        hidden_size=4,
        num_layers=1,
        prediction_horizon=1,
        memory_slots=2,
        attention_hops=1,
        mismatch_threshold=0.5,
    )
    obs = torch.randn(2, 2)
    data = theory_of_mind.ToMInput(
        agent_id="agent",
        char_id="bob",
        observations=obs,
        belief_state={"k1": torch.tensor([1.0, 0.0, 0.0, 0.0])},
    )
    tom.observe(data)
    retrieved = tom.memory.attend("agent", "k1", hops=1)
    assert torch.allclose(retrieved, torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=0.1)


def test_attention_selects_correct_belief_state():
    importlib.reload(theory_of_mind)
    tom = theory_of_mind.activate(
        hidden_size=4,
        num_layers=1,
        prediction_horizon=1,
        memory_slots=4,
        attention_hops=2,
        mismatch_threshold=0.5,
    )
    obs = torch.randn(2, 2)
    data1 = theory_of_mind.ToMInput(
        agent_id="agent",
        char_id="bob",
        observations=obs,
        belief_state={"k1": torch.tensor([1.0, 0.0, 0.0, 0.0])},
    )
    tom.observe(data1)
    data2 = theory_of_mind.ToMInput(
        agent_id="agent",
        char_id="bob",
        observations=obs,
        belief_state={"k2": torch.tensor([0.0, 1.0, 0.0, 0.0])},
    )
    tom.observe(data2)
    retrieved = tom.memory.attend("agent", "k2", hops=2)
    assert retrieved[1] > retrieved[0]
