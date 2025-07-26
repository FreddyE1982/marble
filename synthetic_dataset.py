import numpy as np
from typing import List, Tuple

def generate_sine_wave_dataset(
    n_samples: int = 100,
    *,
    noise_std: float = 0.1,
    seed: int | None = None,
) -> List[Tuple[float, float]]:
    """Return ``(input, target)`` pairs for a noisy sine wave."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(0, 2 * np.pi, n_samples)
    noise = rng.normal(0.0, noise_std, size=n_samples)
    ys = np.sin(xs) + noise
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


def generate_linear_dataset(
    n_samples: int = 100,
    *,
    slope: float = 1.0,
    intercept: float = 0.0,
    noise_std: float = 0.1,
    seed: int | None = None,
) -> List[Tuple[float, float]]:
    """Return noisy linear ``(input, target)`` pairs."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(-1.0, 1.0, n_samples)
    noise = rng.normal(0.0, noise_std, size=n_samples)
    ys = slope * xs + intercept + noise
    return [(float(x), float(y)) for x, y in zip(xs, ys)]
