"""Example usage of the Theory of Mind plugin."""

import torch

import theory_of_mind


if __name__ == "__main__":
    tom = theory_of_mind.activate(
        hidden_size=8,
        num_layers=1,
        prediction_horizon=1,
        memory_slots=8,
        attention_hops=2,
        mismatch_threshold=0.5,
    )
    obs = torch.randn(5, 3)
    data = theory_of_mind.ToMInput(
        agent_id="agent1",
        char_id="alice",
        observations=obs,
        belief_state={"goal": torch.zeros(8)},
    )
    tom.observe(data)
    pred = tom.predict("alice", obs_size=3)
    print("Prediction", pred)
