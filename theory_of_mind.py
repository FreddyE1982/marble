"""Theory of Mind plugin for Marble.

This module models other agents by maintaining recurrent networks per character.
Predictions are broadcast through the global workspace and recorded as markers
in Neuronenblitz for later analysis.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

import global_workspace


class CharacterModel(nn.Module):
    """Simple LSTM-based character model."""

    def __init__(self, obs_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.rnn = nn.LSTM(obs_size, hidden_size, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, obs_size)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(seq)
        return self.head(out[:, -1])


class TheoryOfMind:
    """Maintain models for multiple characters."""

    def __init__(self, nb: object | None, hidden_size: int, num_layers: int, prediction_horizon: int) -> None:
        self.nb = nb
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        self.models: Dict[str, CharacterModel] = {}

    def _get_model(self, char_id: str, obs_size: int) -> CharacterModel:
        if char_id not in self.models:
            self.models[char_id] = CharacterModel(obs_size, self.hidden_size, self.num_layers)
        return self.models[char_id]

    def observe(self, char_id: str, observations: torch.Tensor) -> torch.Tensor:
        """Update internal state with ``observations`` and return prediction."""
        model = self._get_model(char_id, observations.size(-1))
        pred = model(observations.unsqueeze(0))
        error = torch.mean((pred - observations[-1]) ** 2).item()
        if hasattr(self.nb, "log_hot_marker"):
            self.nb.log_hot_marker({"tom_prediction_error": error})
        if global_workspace.workspace is not None:
            global_workspace.workspace.publish(
                "theory_of_mind", {"character": char_id, "prediction": pred.detach()}
            )
        return pred

    def predict(self, char_id: str, obs_size: int) -> torch.Tensor:
        """Return next state prediction without new observation."""
        model = self._get_model(char_id, obs_size)
        state = torch.zeros(1, self.prediction_horizon, obs_size)
        return model(state)


_tom: Optional[TheoryOfMind] = None


def activate(
    nb: object | None = None,
    *,
    hidden_size: int = 16,
    num_layers: int = 1,
    prediction_horizon: int = 1,
) -> TheoryOfMind:
    """Activate the Theory of Mind plugin and attach to ``nb`` if provided."""
    global _tom
    _tom = TheoryOfMind(nb, hidden_size, num_layers, prediction_horizon)
    if nb is not None:
        setattr(nb, "theory_of_mind", _tom)
    return _tom


def get() -> TheoryOfMind | None:
    """Return the active Theory of Mind instance if any."""
    return _tom


def register(*_: object) -> None:
    """Plugin loader compatibility hook."""
    return
