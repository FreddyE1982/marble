import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from marble_core import AttentionModule


def test_attention_weights_sum_to_one():
    am = AttentionModule(temperature=1.0)
    query = np.array([1.0, 0.0])
    keys = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    weights = am.compute(query, keys)
    assert np.allclose(weights.sum(), 1.0)
    assert weights[0] > weights[1]


def test_attention_temperature_effect():
    query = np.array([1.0, 0.0])
    keys = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    high = AttentionModule(temperature=10.0)
    low = AttentionModule(temperature=0.1)
    w_high = high.compute(query, keys)
    w_low = low.compute(query, keys)
    diff_high = w_high[0] - w_high[1]
    diff_low = w_low[0] - w_low[1]
    assert diff_low > diff_high
