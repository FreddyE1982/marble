import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from io import StringIO
import requests
import pandas as pd
from marble_main import MARBLE
from config_loader import load_config


def main() -> None:
    """Train MARBLE on the classic Iris dataset."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    csv_text = requests.get(url, timeout=10).text
    df = pd.read_csv(StringIO(csv_text), header=None)
    records = df.values.tolist()
    examples = [(r[:-1], r[-1]) for r in records if len(r) == 5]
    train = examples[:120]
    val = examples[120:]
    cfg = load_config()
    marble = MARBLE(cfg["core"])
    marble.brain.train(train, epochs=1, validation_examples=val)
    print("Validation loss:", marble.brain.validate(val))


if __name__ == "__main__":
    main()
