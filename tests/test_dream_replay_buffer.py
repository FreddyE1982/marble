import numpy as np

from dream_replay_buffer import DreamExperience, DreamReplayBuffer


def test_eviction_by_salience():
    buf = DreamReplayBuffer(capacity=2, instant_capacity=1)
    low = DreamExperience(0.0, 0.0, reward=0.1, emotion=0.1, arousal=0.1, stress=0.0)
    high = DreamExperience(0.0, 0.0, reward=0.9, emotion=0.9, arousal=0.9, stress=0.0)
    buf.add(low)
    buf.add(high)
    mid = DreamExperience(0.0, 0.0, reward=0.8, emotion=0.8, arousal=0.8, stress=0.0)
    buf.add(mid)
    assert len(buf) == 2
    # low-salience experience should be evicted
    saliences = [e.salience for e in buf.buffer]
    assert min(saliences) >= mid.salience


def test_sampling_bias():
    np.random.seed(0)
    buf = DreamReplayBuffer(capacity=5)
    low = DreamExperience(0, 0, 0.1, 0.1, 0.1, 0.0)
    high = DreamExperience(0, 0, 0.9, 0.9, 0.9, 0.0)
    buf.add(low)
    buf.add(high)
    counts = {"low": 0, "high": 0}
    for _ in range(200):
        sample = buf.sample(1)[0]
        if sample is high:
            counts["high"] += 1
        else:
            counts["low"] += 1
    assert counts["high"] > counts["low"] * 2


def test_weighting_functions():
    sal = np.array([0.2, 0.8], dtype=float)
    buf = DreamReplayBuffer(capacity=3, weighting="quadratic")
    assert np.allclose(buf._apply_weighting(sal), sal**2)
    buf = DreamReplayBuffer(capacity=3, weighting="sqrt")
    assert np.allclose(buf._apply_weighting(sal), np.sqrt(sal))
    buf = DreamReplayBuffer(capacity=3, weighting="uniform")
    assert np.allclose(buf._apply_weighting(sal), np.ones_like(sal))


def test_housekeeping_prunes_low_salience():
    buf = DreamReplayBuffer(capacity=5, instant_capacity=2, housekeeping_threshold=0.5)
    low = DreamExperience(0, 0, 0.1, 0.1, 0.1, 0.0)
    high = DreamExperience(0, 0, 0.9, 0.9, 0.9, 0.0)
    buf.add(low)
    buf.add(high)  # triggers merge + housekeeping
    assert len(buf.buffer) == 1
    assert buf.buffer[0] is high


def test_instant_buffer_merge():
    buf = DreamReplayBuffer(capacity=3, instant_capacity=3)
    exps = [
        DreamExperience(0, 0, 0.2, 0.2, 0.2, 0.0),
        DreamExperience(0, 0, 0.3, 0.3, 0.3, 0.0),
        DreamExperience(0, 0, 0.4, 0.4, 0.4, 0.0),
    ]
    for exp in exps:
        buf.add(exp)
    assert len(buf.buffer) == 3
    assert len(buf.instant_buffer) == 0
