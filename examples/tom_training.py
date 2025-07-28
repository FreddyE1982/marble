"""Example usage of the Theory of Mind plugin."""

import torch

import theory_of_mind


if __name__ == "__main__":
    tom = theory_of_mind.activate(hidden_size=8, num_layers=1, prediction_horizon=1)
    obs = torch.randn(5, 3)
    tom.observe("alice", obs)
    pred = tom.predict("alice", obs_size=3)
    print("Prediction", pred)
