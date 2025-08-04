from __future__ import annotations

class ExperimentTracker:
    """Abstract base class for experiment trackers."""

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log a metrics dictionary for the given step."""
        raise NotImplementedError

    def log_event(self, name: str, data: dict) -> None:
        """Log a structured event with associated payload."""
        raise NotImplementedError

    def finish(self) -> None:
        """Finish the tracking session."""
        pass


class WandbTracker(ExperimentTracker):
    """Weights & Biases experiment tracker."""

    def __init__(self, project: str, entity: str | None = None, run_name: str | None = None) -> None:
        import wandb

        self.run = wandb.init(project=project, entity=entity, name=run_name, reinit=True)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        import wandb

        wandb.log(metrics, step=step)

    def log_event(self, name: str, data: dict) -> None:
        import wandb

        payload = {f"event_{k}": v for k, v in {"name": name, **data}.items()}
        wandb.log(payload)

    def finish(self) -> None:
        import wandb

        wandb.finish()


def attach_tracker_to_events(
    tracker: ExperimentTracker, *, events: list[str] | None = None
) -> callable:
    """Subscribe ``tracker`` to pipeline events.

    Returns a callable that detaches the subscription.
    """
    from event_bus import global_event_bus

    def _callback(name: str, payload: dict) -> None:
        tracker.log_event(name, payload)

    global_event_bus.subscribe(_callback, events=events)

    def _detach() -> None:
        subs = getattr(global_event_bus, "_subscribers", [])
        subs[:] = [s for s in subs if s.get("callback") is not _callback]

    return _detach
