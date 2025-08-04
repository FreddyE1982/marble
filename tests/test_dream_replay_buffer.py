import numpy as np

from dream_replay_buffer import DreamExperience, DreamReplayBuffer


def test_eviction_by_salience():
    buf = DreamReplayBuffer(capacity=2)
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
