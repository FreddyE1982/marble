import pytest

from benchmark_autograd_vs_marble import (
    generate_dataset,
    train_marble,
    train_autograd,
    run_benchmark,
)


def test_generate_dataset_size():
    data = generate_dataset(10, seed=1)
    assert len(data) == 10
    assert isinstance(data[0][0], float)


def test_train_marble_returns_floats():
    data = generate_dataset(20)
    train_data = data[:15]
    val_data = data[15:]
    loss, duration = train_marble(train_data, val_data, epochs=1)
    assert isinstance(loss, float)
    assert isinstance(duration, float)


def test_train_autograd_returns_floats():
    data = generate_dataset(20)
    train_data = data[:15]
    val_data = data[15:]
    loss, duration = train_autograd(train_data, val_data, epochs=1)
    assert isinstance(loss, float)
    assert isinstance(duration, float)


def test_run_benchmark_structure():
    results = run_benchmark()
    assert "marble" in results and "autograd" in results
    assert set(results["marble"].keys()) == {"loss", "time"}
    assert set(results["autograd"].keys()) == {"loss", "time"}
