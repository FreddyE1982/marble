import os
import sys
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from advanced_gpt import kl_divergence, Tensor, train_advanced_gpt, load_text_dataset


def test_kl_divergence_matches_numpy():
    current = Tensor(np.array([[0.5, 1.0]], dtype=np.float32))
    prev = np.array([[0.1, -0.2]], dtype=np.float32)
    loss = kl_divergence(current, prev)
    p = np.exp(current.data - np.max(current.data, axis=-1, keepdims=True))
    p = p / p.sum(axis=-1, keepdims=True)
    q = np.exp(prev - np.max(prev, axis=-1, keepdims=True))
    q = q / q.sum(axis=-1, keepdims=True)
    ref = np.sum(p * (np.log(p + 1e-8) - np.log(q + 1e-8)))
    assert np.isclose(loss.data, ref)


def test_training_uses_previous_logits(tmp_path):
    txt = tmp_path / "tiny.txt"
    txt.write_text("abcdefg" * 2)
    data, vocab = load_text_dataset(str(txt), vocab_size=10, block_size=3)
    _, losses, kls = train_advanced_gpt(
        data,
        vocab_size=len(vocab),
        block_size=3,
        num_layers=1,
        num_heads=1,
        hidden_dim=16,
        epochs=2,
        lr=0.05,
        batch_size=2,
        seed=0,
        distill_alpha=0.5,
        logits_path=str(tmp_path / "logits.pkl"),
    )
    assert len(kls) == 2
    assert kls[0] == 0.0
    assert kls[1] > 0.0
    with open(tmp_path / "logits.pkl", "rb") as f:
        logs = pickle.load(f)
    assert len(logs) == 2 and logs[1]["epoch"] == 1
