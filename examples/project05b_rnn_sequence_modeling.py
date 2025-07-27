import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from io import StringIO
import requests
import numpy as np
from marble_main import MARBLE
from config_loader import load_config


def main() -> None:
    """Train a simple RNN on a character sequence dataset."""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = requests.get(url, timeout=10).text[:1000]
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    examples = []
    for i in range(len(text) - 10):
        seq = [char_to_idx[c] for c in text[i : i + 10]]
        target = char_to_idx[text[i + 10]]
        examples.append((seq, target))
    train = examples[:500]
    cfg = load_config()
    marble = MARBLE(cfg["core"])
    marble.brain.train(train, epochs=1)
    print("Trained on", len(train), "sequences")


if __name__ == "__main__":
    main()
