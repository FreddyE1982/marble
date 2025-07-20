import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gpt_training import generate_dataset, train_gpt


def test_dataset_generation():
    data = generate_dataset(vocab_size=10, num_samples=5, block_size=4, seed=0)
    assert len(data) == 5
    assert data[0].shape[0] == 5


def test_gpt_training_reduces_loss():
    data = generate_dataset(vocab_size=20, num_samples=8, block_size=4, seed=0)
    _, losses = train_gpt(
        data,
        vocab_size=20,
        block_size=4,
        num_layers=1,
        num_heads=2,
        hidden_dim=32,
        epochs=3,
        lr=0.01,
        seed=0,
    )
    assert losses[-1] <= losses[0] + 1e-9


def test_gpt_training_gradient_clipping():
    data = generate_dataset(vocab_size=15, num_samples=3, block_size=4, seed=1)
    _, _, norms = train_gpt(
        data,
        vocab_size=15,
        block_size=4,
        num_layers=1,
        num_heads=1,
        hidden_dim=16,
        epochs=1,
        lr=0.5,
        seed=1,
        max_grad_norm=0.2,
        return_grad_norms=True,
    )
    assert norms and all(n <= 0.2 + 1e-6 for n in norms)
