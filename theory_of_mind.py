"""Theory of Mind plugin for Marble.

This module models other agents by maintaining recurrent networks per character.
Predictions are broadcast through the global workspace and recorded as markers
in Neuronenblitz for later analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import nn

import global_workspace


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------


@dataclass
class ToMInput:
    """Input structure for :class:`TheoryOfMind.observe`.

    Attributes:
        agent_id: Identifier of the observing agent.
        char_id: Target character identifier whose state is being tracked.
        observations: Sequence of observations for the character.
        belief_state: Mapping of belief keys to value tensors.
    """

    agent_id: str
    char_id: str
    observations: torch.Tensor
    belief_state: Dict[str, torch.Tensor]

    def validate(self) -> None:
        if not isinstance(self.agent_id, str) or not self.agent_id:
            raise ValueError("agent_id must be a non-empty string")
        if not isinstance(self.char_id, str) or not self.char_id:
            raise ValueError("char_id must be a non-empty string")
        if not isinstance(self.observations, torch.Tensor):
            raise TypeError("observations must be a tensor")
        if self.observations.dim() != 2:
            raise ValueError("observations must be a 2D tensor [time, features]")
        if not isinstance(self.belief_state, dict):
            raise TypeError("belief_state must be a dictionary")
        for k, v in self.belief_state.items():
            if not isinstance(k, str):
                raise TypeError("belief_state keys must be strings")
            if not isinstance(v, torch.Tensor):
                raise TypeError("belief_state values must be tensors")

    def to_dict(self) -> Dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "char_id": self.char_id,
            "observations": self.observations.tolist(),
            "belief_state": {k: v.tolist() for k, v in self.belief_state.items()},
        }

    @staticmethod
    def from_dict(data: Dict[str, object]) -> "ToMInput":
        observations = torch.tensor(data["observations"], dtype=torch.float32)
        belief_state = {
            k: torch.tensor(v, dtype=torch.float32)
            for k, v in (data.get("belief_state") or {}).items()
        }
        obj = ToMInput(
            agent_id=str(data["agent_id"]),
            char_id=str(data["char_id"]),
            observations=observations,
            belief_state=belief_state,
        )
        obj.validate()
        return obj


# ---------------------------------------------------------------------------
# Belief memory with multi-hop attention
# ---------------------------------------------------------------------------


def _string_to_vec(text: str, dim: int) -> torch.Tensor:
    vec = torch.zeros(dim, dtype=torch.float32)
    data = text.encode("utf-8")
    for i, ch in enumerate(data):
        vec[i % dim] += ch / 255.0
    return vec


class ToMModule(nn.Module):
    """Memory module storing beliefs as key-value slots."""

    def __init__(self, hidden_size: int, capacity: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.capacity = capacity
        self.keys = nn.Parameter(torch.zeros(capacity, hidden_size))
        self.values = nn.Parameter(torch.zeros(capacity, hidden_size))
        self.register_buffer("_next", torch.tensor(0, dtype=torch.long))

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------
    def _encode_key(self, agent_id: str, key: str) -> torch.Tensor:
        half = self.hidden_size // 2
        aid = _string_to_vec(agent_id, half)
        kvec = _string_to_vec(key, self.hidden_size - half)
        return torch.cat([aid, kvec], dim=0)

    def _encode_value(self, value: torch.Tensor) -> torch.Tensor:
        if value.numel() != self.hidden_size:
            v = torch.zeros(self.hidden_size, dtype=torch.float32)
            flat = value.flatten().to(torch.float32)
            size = min(flat.numel(), self.hidden_size)
            v[:size] = flat[:size]
            return v
        return value.to(torch.float32).view(self.hidden_size)

    # ------------------------------------------------------------------
    # Memory operations
    # ------------------------------------------------------------------
    def write(self, agent_id: str, belief_state: Dict[str, torch.Tensor]) -> None:
        for key, val in belief_state.items():
            slot = int(self._next.item())
            self.keys.data[slot] = self._encode_key(agent_id, key)
            self.values.data[slot] = self._encode_value(val)
            self._next.add_(1)
            if self._next >= self.capacity:
                self._next.zero_()

    def attend(self, agent_id: str, key: str, hops: int) -> torch.Tensor:
        query = self._encode_key(agent_id, key)
        for _ in range(max(1, hops)):
            weights = torch.softmax(self.keys @ query, dim=0)
            read = weights @ self.values
            query = query + read
        return read


# ---------------------------------------------------------------------------
# Character model and Theory of Mind container
# ---------------------------------------------------------------------------


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
    """Maintain models for multiple characters and belief memories."""

    def __init__(
        self,
        nb: object | None,
        hidden_size: int,
        num_layers: int,
        prediction_horizon: int,
        memory_slots: int,
        attention_hops: int,
        mismatch_threshold: float,
    ) -> None:
        self.nb = nb
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        self.memory = ToMModule(hidden_size, memory_slots)
        self.attention_hops = attention_hops
        self.mismatch_threshold = mismatch_threshold
        self.mismatches: List[dict] = []
        self.models: Dict[str, CharacterModel] = {}

    def _get_model(self, char_id: str, obs_size: int) -> CharacterModel:
        if char_id not in self.models:
            self.models[char_id] = CharacterModel(obs_size, self.hidden_size, self.num_layers)
        return self.models[char_id]

    def observe(self, data: ToMInput) -> torch.Tensor:
        """Update internal state with observations and belief state.

        Args:
            data: :class:`ToMInput` containing agent, character and belief info.
        """

        data.validate()
        model = self._get_model(data.char_id, data.observations.size(-1))
        pred = model(data.observations.unsqueeze(0))

        # Store beliefs and compute mismatches
        self.memory.write(data.agent_id, data.belief_state)
        for key, value in data.belief_state.items():
            retrieved = self.memory.attend(data.agent_id, key, self.attention_hops)
            mismatch = torch.mean((retrieved - value) ** 2).item()
            if mismatch > self.mismatch_threshold:
                self.mismatches.append(
                    {
                        "agent_id": data.agent_id,
                        "character": data.char_id,
                        "belief_key": key,
                        "mismatch": mismatch,
                    }
                )

        error = torch.mean((pred - data.observations[-1]) ** 2).item()
        if hasattr(self.nb, "log_hot_marker"):
            self.nb.log_hot_marker({"tom_prediction_error": error})
        if global_workspace.workspace is not None:
            global_workspace.workspace.publish(
                "theory_of_mind",
                {
                    "agent": data.agent_id,
                    "character": data.char_id,
                    "prediction": pred.detach(),
                },
            )
        return pred

    def get_mismatches(self) -> List[dict]:
        """Return accumulated belief mismatches."""
        return list(self.mismatches)

    def save_mismatches(self, path: str) -> None:
        """Serialise mismatches to ``path`` in JSON format."""
        import json

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.mismatches, fh, indent=2)

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
    memory_slots: int = 16,
    attention_hops: int = 1,
    mismatch_threshold: float = 0.1,
) -> TheoryOfMind:
    """Activate the Theory of Mind plugin and attach to ``nb`` if provided."""
    global _tom
    _tom = TheoryOfMind(
        nb,
        hidden_size,
        num_layers,
        prediction_horizon,
        memory_slots,
        attention_hops,
        mismatch_threshold,
    )
    if nb is not None:
        setattr(nb, "theory_of_mind", _tom)
    return _tom


def get() -> TheoryOfMind | None:
    """Return the active Theory of Mind instance if any."""
    return _tom


def register(*_: object) -> None:
    """Plugin loader compatibility hook."""
    return
