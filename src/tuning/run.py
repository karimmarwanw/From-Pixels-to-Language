"""Run a grid or random search: one ordinary, W&B-tracked training run per trial.

For every trial from src/tuning/search.py:
    base config + fixed overrides + the trial's settings + extra CLI overrides
    -> train_classification(cfg, group=<experiment>) -> one row in results.csv

    - The dataset is loaded once and shared by all trials (no re-reading 1.1 GB per trial).
    - results.csv (<results_dir>/<experiment>/) gets a row after EVERY trial, so a crash or a
      closed laptop loses at most the trial that was running. Re-running the same command skips
      trials that finished ok and tries failed ones again.
    - A trial that crashes (e.g. the loss becomes NaN, or a download fails) is recorded as
      "failed" and the search goes on. Its W&B run is closed as failed and its row stays in
      results.csv even after a successful retry, so every attempt shows up for the report.
    - Trial checkpoints are deleted afterwards unless keep_checkpoints: true. A ResNet50 stage 2
      best.pt + last.pt is ~550 MB, and the winner is retrained at full length anyway.
      history.csv stays for the curves.
    - wall_time_s = the whole trial (model build, W&B, training, checkpoints), the time used for
      the grid vs random comparison. train_time_s = the training epochs only.
"""

import csv
import gc
import logging
import time
from collections.abc import Callable
from pathlib import Path

import torch

from src.tuning.search import make_trials, trial_name, trial_overrides
from src.utils.config import TrainConfig, TuneConfig, load_train_config

logger = logging.getLogger(__name__)

RESERVED_KEYS = {"experiment", "run_name"}  # set per trial by the search itself


def results_file(cfg: TuneConfig, results_dir: str | Path | None = None) -> Path:
    """<results_dir>/<experiment>/results.csv (results_dir defaults to the one in the tuning config)."""
    return Path(results_dir or cfg.results_dir) / cfg.experiment / "results.csv"


def read_results(path: str | Path) -> list[dict[str, str]]:
    """Rows of a results.csv (empty list if the search has not started yet)."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _check_extra_overrides(cfg: TuneConfig, extra: list[str]) -> None:
    """Extra CLI overrides apply to every trial, so they must not touch the settings being searched."""
    for item in extra:
        key = item.split("=", 1)[0]
        if key in cfg.space or key in RESERVED_KEYS:
            raise ValueError(f"--override {key} would overwrite a setting that the search itself sets per trial")


def _delete_checkpoints(checkpoint_dir: str | Path, run_name: str) -> None:
    """Remove the .pt files of one trial (every <run_name>_<timestamp>/ folder), keep history.csv."""
    for run_dir in Path(checkpoint_dir).glob(f"{run_name}_*"):
        for pt in run_dir.glob("*.pt"):
            pt.unlink()


def _free_memory() -> None:
    """Release the finished trial's model and optimizer before the next one is built."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def run_search(
    cfg: TuneConfig,
    extra_overrides: list[str] | None = None,
    max_trials: int | None = None,
    results_dir: str | Path | None = None,
    hf_train=None,
    train_fn: Callable[..., dict] | None = None,
) -> Path:
    """Run every trial of a search that is not in results.csv yet. Returns the path of results.csv.

    Args:
        extra_overrides: training-config overrides for EVERY trial, e.g. ["model.init_from=.../best.pt"].
        max_trials: run only the first N trials (1 = check the setup before the full search).
        results_dir: replaces cfg.results_dir (e.g. a separate folder for smoke tests).
        hf_train: pre-loaded train split (tests); loaded once from the Hub if None.
        train_fn: replaces train_classification (tests only).
    """
    extra = list(extra_overrides or [])
    _check_extra_overrides(cfg, extra)
    if train_fn is None:
        from src.training.run import train_classification as train_fn

    trials = make_trials(cfg)[:max_trials] if max_trials else make_trials(cfg)
    out = results_file(cfg, results_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {row["run_name"] for row in read_results(out) if row["status"] == "ok"}  # failed ones are retried

    base: TrainConfig = load_train_config(cfg.base_config, trial_overrides(cfg.fixed) + extra)
    metric = f"best_{base.train.monitor}"
    columns = ["trial", "run_name", "status", *cfg.space, metric, "best_epoch", "epochs_run",
               "train_time_s", "wall_time_s", "run_id", "error"]
    if hf_train is None and len(done) < len(trials):
        from src.data.loading import load_cub_split

        hf_train = load_cub_split(base.data.train_name, "train", base.data.cache_dir)

    logger.info("%s search %s: %d trials, %d already in %s", cfg.method, cfg.experiment, len(trials),
                len(done & {trial_name(cfg.method, i, t) for i, t in enumerate(trials, 1)}), out)
    for number, params in enumerate(trials, 1):
        name = trial_name(cfg.method, number, params)
        if name in done:
            logger.info("Trial %d/%d %s: already done, skipped", number, len(trials), name)
            continue
        overrides = (trial_overrides(cfg.fixed) + trial_overrides(params)
                     + [f"experiment={cfg.experiment}", f"run_name={name}"] + extra)
        train_cfg = load_train_config(cfg.base_config, overrides)
        logger.info("Trial %d/%d %s: %s", number, len(trials), name, " ".join(trial_overrides(params)))

        row: dict[str, object] = {"trial": number, "run_name": name, **params}
        t0 = time.perf_counter()
        try:
            result = train_fn(train_cfg, group=cfg.experiment, hf_train=hf_train)
            row.update({"status": "ok", metric: result["best_metric"], "best_epoch": result["best_epoch"],
                        "epochs_run": result["epochs_run"], "train_time_s": result["train_time_s"],
                        "run_id": result["run_id"]})
        except Exception as e:  # keep searching; the failure itself is a result for the report
            logger.exception("Trial %s failed", name)
            row.update({"status": "failed", "error": f"{type(e).__name__}: {e}"})
        row["wall_time_s"] = time.perf_counter() - t0

        new_file = not out.exists()
        with open(out, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        if not cfg.keep_checkpoints:
            _delete_checkpoints(train_cfg.train.checkpoint_dir, name)
        _free_memory()
    return out


def summarize(path: str | Path, metric: str = "best_val_acc", mode: str = "max") -> dict:
    """Best trial, trial counts and total time of one search, from its results.csv.

    A trial that failed and was retried has two rows: it counts once, as ok if the retry worked.
    total_time_s includes the failed attempts, because they used real time too.
    """
    rows = read_results(path)
    ok = [r for r in rows if r["status"] == "ok"]
    n_trials = len({r["run_name"] for r in rows})
    best = (max if mode == "max" else min)(ok, key=lambda r: float(r[metric])) if ok else None
    return {
        "n_trials": n_trials, "n_ok": len(ok), "n_failed": n_trials - len(ok), "n_attempts": len(rows),
        "total_time_s": sum(float(r["wall_time_s"]) for r in rows), "best": best,
    }
