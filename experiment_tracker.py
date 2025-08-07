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


class KuzuExperimentTracker(ExperimentTracker):
    """Persist metrics and events inside a Kùzu graph database."""

    def __init__(self, db_path: str) -> None:
        from kuzu_interface import KuzuGraphDatabase

        self.db = KuzuGraphDatabase(db_path)
        self._metric_id = 0
        self._event_id = 0
        self._init_schema()

    def _init_schema(self) -> None:
        try:
            self.db.create_node_table(
                "Metric",
                {
                    "id": "INT64",
                    "name": "STRING",
                    "step": "INT64",
                    "value": "DOUBLE",
                    "timestamp": "TIMESTAMP",
                },
                "id",
            )
        except Exception:
            pass
        try:
            self.db.create_node_table(
                "Event",
                {
                    "id": "INT64",
                    "name": "STRING",
                    "step": "INT64",
                    "payload": "STRING",
                    "timestamp": "TIMESTAMP",
                },
                "id",
            )
        except Exception:
            pass

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        import datetime as _dt
        import torch

        ts = _dt.datetime.now(_dt.timezone.utc)
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = float(value.detach().cpu())
            else:
                value = float(value)
            self._metric_id += 1
            self.db.add_node(
                "Metric",
                {
                    "id": self._metric_id,
                    "name": name,
                    "step": step,
                    "value": value,
                    "timestamp": ts,
                },
            )

    def log_event(self, name: str, data: dict) -> None:
        import datetime as _dt
        import json

        ts = _dt.datetime.now(_dt.timezone.utc)
        self._event_id += 1
        self.db.add_node(
            "Event",
            {
                "id": self._event_id,
                "name": name,
                "step": int(data.get("index", data.get("step", 0))),
                "payload": json.dumps(data, default=str),
                "timestamp": ts,
            },
        )

    def finish(self) -> None:
        self.db.close()


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
