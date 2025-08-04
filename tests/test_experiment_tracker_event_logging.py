from __future__ import annotations

import torch

from event_bus import PROGRESS_EVENT
from experiment_tracker import (
    ExperimentTracker,
    attach_tracker_to_events,
)
from pipeline import Pipeline


class DummyTracker(ExperimentTracker):
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        pass

    def log_event(self, name: str, data: dict) -> None:
        self.events.append((name, data))

    def finish(self) -> None:
        pass


def sample_step(device: str) -> int:
    return torch.tensor([1], device=device).sum().item()


def test_tracker_receives_pipeline_events() -> None:
    tracker = DummyTracker()
    detach = attach_tracker_to_events(tracker, events=[PROGRESS_EVENT])
    pipe = Pipeline()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.add_step(func="sample_step", module=__name__, params={"device": dev})
    pipe.execute()
    detach()
    assert any(evt[0] == PROGRESS_EVENT for evt in tracker.events)
    assert tracker.events[0][1]["device"] == (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
