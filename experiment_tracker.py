from __future__ import annotations

class ExperimentTracker:
    """Abstract base class for experiment trackers."""

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log a metrics dictionary for the given step."""
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

    def finish(self) -> None:
        import wandb

        wandb.finish()
