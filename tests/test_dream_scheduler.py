from dream_replay_buffer import DreamReplayBuffer, DreamExperience
from dream_scheduler import DreamScheduler


class DummyLearner:
    def __init__(self) -> None:
        self.seen = []

    def train_example(self, inp: float, tgt: float) -> None:  # pragma: no cover - simple
        self.seen.append((inp, tgt))


def test_scheduler_replays_and_prunes():
    buffer = DreamReplayBuffer(
        capacity=10,
        weighting="linear",
        instant_capacity=10,
        housekeeping_threshold=0.1,
    )
    high = DreamExperience(1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
    low = DreamExperience(1.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    buffer.add(high)
    buffer.add(low)

    learner = DummyLearner()
    scheduler = DreamScheduler(learner, buffer, batch_size=5)
    replayed = scheduler.replay()

    assert replayed == 1
    assert learner.seen == [(1.0, 1.0)]
    assert len(buffer.buffer) == 1  # low-salience experience pruned
