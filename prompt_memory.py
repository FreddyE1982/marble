import json
from collections import deque
from time import time
from typing import Deque, Dict, List, Tuple


class PromptMemory:
    """Caches recent (input, output) pairs for in-context learning.

    Parameters
    ----------
    max_size: int
        Maximum number of pairs to keep. Oldest pairs are evicted first.
    """

    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self._data: Deque[Dict[str, str]] = deque(maxlen=max_size)

    def add(self, inp: str, out: str) -> None:
        """Add a new `(input, output)` pair to the memory."""
        self._data.append({"input": inp, "output": out, "timestamp": time()})

    def get_pairs(self) -> List[Tuple[str, str]]:
        """Return stored `(input, output)` pairs ignoring timestamps."""
        return [(item["input"], item["output"]) for item in list(self._data)]

    def get_records(self) -> List[Dict[str, str]]:
        """Return full records including timestamps."""
        return list(self._data)

    def get_prompt(self) -> str:
        """Concatenate stored pairs into a prompt string."""
        segments = []
        for pair in self._data:
            segments.append(f"Input: {pair['input']}\nOutput: {pair['output']}")
        return "\n".join(segments)

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
                data: List[Dict[str, str]] = json.load(f)
            for pair in data[-max_size:]:
                memory._data.append(pair)
        except FileNotFoundError:
            pass
        return memory
