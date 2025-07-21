import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import requests
from io import StringIO
from marble_main import MARBLE
from config_loader import load_config

def main() -> None:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    csv_text = requests.get(url, timeout=10).text
    df = pd.read_csv(StringIO(csv_text), sep=";")
    records = df.values.tolist()
    examples = [
        (float(sum(r[:-1])) / len(r[:-1]), float(r[-1]))
        for r in records
    ]
    train = examples[:150]
    val = examples[150:200]
    cfg = load_config()
    marble = MARBLE(cfg["core"])
    marble.brain.train(train, epochs=1, validation_examples=val)
    print("Validation loss:", marble.brain.validate(val))

if __name__ == "__main__":
    main()
