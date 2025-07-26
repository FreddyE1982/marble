import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from synthetic_dataset import generate_sine_wave_dataset, generate_linear_dataset


def test_sine_dataset_deterministic():
    data1 = generate_sine_wave_dataset(10, seed=42)
    data2 = generate_sine_wave_dataset(10, seed=42)
    assert data1 == data2


def test_linear_dataset_deterministic():
    data1 = generate_linear_dataset(5, slope=2.0, intercept=1.0, seed=123)
    data2 = generate_linear_dataset(5, slope=2.0, intercept=1.0, seed=123)
    assert data1 == data2
