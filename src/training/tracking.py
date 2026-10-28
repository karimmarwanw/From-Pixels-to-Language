"""Experiment tracking with Weights & Biases (W&B, part of CoreWeave Forge), behind a small wrapper.

The Trainer never imports wandb: it only calls log_metrics(step) on a tracker, so the tracking
tool can change without touching the training loop.

Where runs go: entity (team) / project from the config, e.g. cub-dl-team/from-pixels-to-language,
so every member's runs (Kaggle, laptops) land on ONE shared online dashboard.
    - name   = run_name, says what changed (scratch-cnn-baseline, resnet50-ft-lr3e-4)
    - group  = experiment (cls-scratch, cls-transfer), or one tuning search (cls-tuning-grid-...)
    - config = the flattened YAML config + environment (git commit, library versions, GPU)
    - history = per-epoch metrics; summary = best metric, test numbers, training time
Login: `wandb login` once on a laptop, or the WANDB_API_KEY secret on Kaggle. Never commit the key.
"""

import logging
import os
import platform
import subprocess
from pathlib import Path

import torch
import wandb

logger = logging.getLogger(__name__)


def git_commit() -> str:
    """Current git commit hash (+ "-dirty" if there are uncommitted changes), or "unknown"."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def environment_tags() -> dict[str, str]:
    """What a reader needs to reproduce the run: code version, library versions, hardware."""
    import torchvision

    return {
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "wandb": wandb.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    }


class WandbTracker:
    """Context manager around one W&B run.

    with WandbTracker("from-pixels-to-language", "cub-dl-team", "cls-scratch", "scratch-cnn-baseline") as t:
        t.log_params({...}); t.log_metrics({"val_acc": 0.3}, step=1); t.log_summary({"best_val_acc": 0.3})
    """

    def __init__(
        self, project: str, entity: str | None, experiment: str, run_name: str,
        run_id: str | None = None, group: str | None = None, mode: str = "online",
    ) -> None:
        """
        Args:
            entity: W&B team (or user); None = your default entity.
            run_id: set = reopen an existing run (resume after a Kaggle disconnect, or add test results).
            group: tuning search name; trials of one search share it. None = the experiment name.
            mode: "online" (shared dashboard), "offline" (saved locally, `wandb sync` later), "disabled".
        """
        self.project, self.entity, self.experiment = project, entity, experiment
        self.run_name, self.run_id, self.group, self.mode = run_name, run_id, group, mode
        self.run = None

    def __enter__(self) -> "WandbTracker":
        if self.mode == "online" and not (os.environ.get("WANDB_API_KEY") or wandb.login(prompt=False)):
            raise RuntimeError("Not logged in to W&B: run `wandb login`, or set the WANDB_API_KEY secret on Kaggle "
                               "(or use --override tracking.wandb_mode=offline)")
        self.run = wandb.init(
            project=self.project, entity=self.entity, name=self.run_name, group=self.group or self.experiment,
            tags=[self.experiment], id=self.run_id, resume="must" if self.run_id else None, mode=self.mode,
        )
        self.run_id = self.run.id
        logger.info("W&B run %s (%s / %s) %s", self.run_id, self.experiment, self.run_name, self.run.url or "")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.run.finish(exit_code=1 if exc_type else 0)

    def log_params(self, params: dict[str, object]) -> None:
        self.run.config.update(params, allow_val_change=True)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Per-epoch values: they become the curves on the dashboard (x-axis = step = epoch)."""
        self.run.log({k: float(v) for k, v in metrics.items()}, step=step)

    def log_summary(self, metrics: dict[str, float]) -> None:
        """One final value per key (best metric, test accuracy): the columns of the runs table."""
        self.run.summary.update({k: float(v) for k, v in metrics.items()})

    def set_tags(self, tags: dict[str, str]) -> None:
        self.run.config.update({f"env.{k}": v for k, v in tags.items()}, allow_val_change=True)

    def log_artifact(self, path: str | Path, kind: str = "file") -> None:
        """Upload a file to the run (history.csv, a figure; kind="model" for checkpoints)."""
        path = Path(path)
        artifact = wandb.Artifact(name=f"{self.run_name}-{path.stem}", type=kind)
        artifact.add_file(str(path))
        self.run.log_artifact(artifact)
