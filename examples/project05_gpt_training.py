import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import requests
import tempfile
from advanced_gpt import load_text_dataset, train_advanced_gpt


def main() -> None:
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    path = tempfile.gettempdir() + "/tinyshakespeare.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(requests.get(url, timeout=10).text)
    data, vocab = load_text_dataset(path, vocab_size=20, block_size=32)
    model, losses = train_advanced_gpt(
        data[:50],
        vocab_size=len(vocab),
        block_size=32,
        num_layers=1,
        num_heads=1,
        hidden_dim=32,
        epochs=1,
        batch_size=2,
    )
    print("Loss:", losses[-1])


if __name__ == "__main__":
    main()
