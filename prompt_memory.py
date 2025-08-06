import json
from collections import deque
from time import time
from typing import Any, Deque, Dict, List, Tuple


class PromptMemory:
    """Caches recent (input, output) pairs for in-context learning.

    Parameters
    ----------
    max_size: int
        Maximum number of pairs to keep. Oldest pairs are evicted first.
    """

    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self._data: Deque[Dict[str, Any]] = deque(maxlen=max_size)

    def __len__(self) -> int:
        """Return the number of stored records."""
        return len(self._data)

    def add(self, inp: str, out: str) -> None:
        """Add a new `(input, output)` pair to the memory."""
        self._data.append({"input": inp, "output": out, "timestamp": time()})

    def get_pairs(self) -> List[Tuple[str, str]]:
        """Return stored `(input, output)` pairs ignoring timestamps."""
        return [(item["input"], item["output"]) for item in list(self._data)]

    def get_records(self) -> List[Dict[str, Any]]:
        """Return full records sorted by ``timestamp``."""
        return sorted(self._data, key=lambda r: r["timestamp"])

    def get_prompt(self) -> str:
        """Concatenate stored pairs into a prompt string."""
        segments = []
        for pair in self._data:
            segments.append(f"Input: {pair['input']}\nOutput: {pair['output']}")
        return "\n".join(segments)

    def composite_with(self, new_input: str, max_chars: int = 2048) -> str:
        """Return ``prompt + new_input`` truncated to ``max_chars`` characters.

        Oldest records are dropped until the composite fits within the limit.
        If no records are stored, ``new_input`` is returned unchanged.
        """
        pairs = list(self._data)
        composite = new_input
        if pairs:
            prompt = self.get_prompt()
            composite = f"{prompt}\nInput: {new_input}" if prompt else new_input
            # Drop oldest pairs until within size limit
            while len(composite) > max_chars and pairs:
                pairs.pop(0)
                prompt = "\n".join(
                    f"Input: {p['input']}\nOutput: {p['output']}" for p in pairs
                )
                composite = f"{prompt}\nInput: {new_input}" if prompt else new_input
        # Ensure final string is not longer than max_chars
        if len(composite) > max_chars:
            composite = composite[-max_chars:]
        return composite

    def serialize(self, path: str) -> None:
        """Save stored pairs to ``path`` in JSON format."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(list(self._data), f)

    @classmethod
    def load(cls, path: str, max_size: int = 10) -> "PromptMemory":
        """Load stored pairs from ``path``.

        Parameters
        ----------
        path: str
            JSON file previously produced by :meth:`serialize`.
        max_size: int
            Maximum number of pairs to keep in the loaded memory.
        """
        memory = cls(max_size=max_size)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data: List[Dict[str, Any]] = json.load(f)
            data = sorted(data, key=lambda r: r.get("timestamp", 0))
            for pair in data[-max_size:]:
                memory._data.append(pair)
        except FileNotFoundError:
            pass
        return memory
