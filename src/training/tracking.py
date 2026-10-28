"""Experiment tracking with MLflow, behind a small wrapper so the Trainer never imports mlflow.

Backend: a local SQLite file (`sqlite:///mlflow.db`) with artifacts in ./mlruns. MLflow 3.x
refuses the old plain-folder backend (`./mlruns` as tracking URI) unless
MLFLOW_ALLOW_FILE_STORE=true is set, so we use the backend MLflow recommends.
View runs with:  mlflow ui --backend-store-uri sqlite:///mlflow.db

Each run logs: the flattened config (params), per-epoch metrics, tags with the git commit,
library versions and GPU, the checkpoint paths, and artifacts (history CSV, figures).
Tuning (Karim) opens one parent run per search and passes nested=True for each trial.
"""

import logging
import os
import platform
import subprocess
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # silence an mlflow startup message

import mlflow  # noqa: E402
import torch  # noqa: E402

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
        "mlflow": mlflow.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    }


class MLflowTracker:
    """Context manager around one MLflow run.

    with MLflowTracker("sqlite:///mlflow.db", "cls-scratch", "scratch-cnn-baseline") as t:
        t.log_params({...}); t.log_metrics({"val_acc": 0.3}, step=1)
    """

    def __init__(
        self, tracking_uri: str, experiment: str, run_name: str,
        run_id: str | None = None, nested: bool = False,
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment = experiment
        self.run_name = run_name
        self.run_id = run_id  # set: reopen an existing run (resume after a Kaggle/Colab disconnect)
        self.nested = nested

    def __enter__(self) -> "MLflowTracker":
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment)
        run = mlflow.start_run(run_id=self.run_id, run_name=None if self.run_id else self.run_name,
                               nested=self.nested)
        self.run_id = run.info.run_id
        logger.info("MLflow run %s (%s / %s)", self.run_id, self.experiment, self.run_name)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        mlflow.end_run(status="FAILED" if exc_type else "FINISHED")

    def log_params(self, params: dict[str, object]) -> None:
        mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def set_tags(self, tags: dict[str, str]) -> None:
        mlflow.set_tags(tags)

    def log_artifact(self, path: str | Path, subdir: str | None = None) -> None:
        mlflow.log_artifact(str(path), artifact_path=subdir)
