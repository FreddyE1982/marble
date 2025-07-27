import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.linear_model import LinearRegression

from config_loader import load_config
from marble_main import MARBLE


def main() -> None:
    rng = np.random.RandomState(0)
    X = rng.rand(200, 3)
    y = X @ np.array([1.0, 2.0, -1.5]) + 0.1 * rng.randn(200)
    reg = LinearRegression().fit(X, y)
    preds = reg.predict(X)
    examples = [(float(x.mean()), float(t)) for x, t in zip(X, preds)]

    cfg = load_config()
    marble = MARBLE(cfg["core"])
    marble.brain.train(examples, epochs=1)
    print("Trained with sklearn-generated labels")


if __name__ == "__main__":
    main()
