import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from advanced_gpt import load_text_dataset, train_advanced_gpt


def test_load_text_dataset(tmp_path):
    txt = tmp_path / "tiny.txt"
    txt.write_text("hello world")
    data, vocab = load_text_dataset(str(txt), vocab_size=20, block_size=3)
    assert data
    assert isinstance(vocab, dict)
    assert data[0].shape[0] == 4


def test_train_advanced_gpt_reduces_loss(tmp_path):
    txt = tmp_path / "tiny.txt"
    txt.write_text("abcdefg" * 2)
    data, vocab = load_text_dataset(str(txt), vocab_size=10, block_size=3)
    _, losses = train_advanced_gpt(
        data,
        vocab_size=len(vocab),
        block_size=3,
        num_layers=1,
        num_heads=1,
        hidden_dim=16,
        epochs=3,
        lr=0.05,
        batch_size=2,
        seed=0,
    )
    assert losses[-1] <= losses[0]


def test_train_advanced_gpt_gradient_clipping(tmp_path):
    txt = tmp_path / "tiny.txt"
    txt.write_text("abcdefgh" * 2)
    data, vocab = load_text_dataset(str(txt), vocab_size=10, block_size=3)
    _, _, norms = train_advanced_gpt(
        data,
        vocab_size=len(vocab),
        block_size=3,
        num_layers=1,
        num_heads=1,
        hidden_dim=16,
        epochs=1,
        lr=0.2,
        batch_size=2,
        seed=1,
        max_grad_norm=0.25,
        return_grad_norms=True,
    )
    assert norms and all(n <= 0.25 + 1e-6 for n in norms)
