"""One Trainer for every task: it only needs a model, a loss, an optimizer, metric functions and loaders.

Each epoch: train over the train loader -> evaluate on the val loader -> log -> checkpoint.
    - Mixed precision (CUDA only): forward pass in float16 where safe, GradScaler keeps small
      gradients from underflowing to 0. Roughly 2x faster on a T4 and less memory.
    - Checkpoints: last.pt every epoch (resume after a disconnect), best.pt when the monitored
      validation metric improves (model selection on val, never test).
    - Early stopping: stop after `patience` epochs without improvement on the val metric.
Nothing here is classification-specific; the task decides loss_fn and metric_fns.
"""

import csv
import logging
import math
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.utils.config import TrainingConfig

logger = logging.getLogger(__name__)

MetricFn = Callable[[Tensor, Tensor], float]  # (outputs, targets) -> batch mean


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        cfg: TrainingConfig,
        ckpt_dir: str | Path,
        scheduler=None,
        metric_fns: dict[str, MetricFn] | None = None,
        tracker=None,
        extra_state: dict | None = None,
    ) -> None:
        """
        Args:
            tracker: anything with log_metrics(dict, step); None = no tracking (tests only).
            extra_state: stored in every checkpoint (config, class names, W&B run id, ...).
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.cfg = cfg
        self.ckpt_dir = Path(ckpt_dir)
        self.scheduler = scheduler
        self.metric_fns = metric_fns or {}
        self.tracker = tracker
        self.extra_state = extra_state or {}

        self.use_amp = cfg.amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.sign = 1 if cfg.monitor_mode == "max" else -1  # compare as "bigger is better"
        self.start_epoch = 0
        self.best_metric = -math.inf
        self.best_epoch = -1
        self.history: list[dict[str, float]] = []

    # ------------------------------------------------------------------ loops
    def _run_epoch(self, loader: DataLoader, train: bool) -> dict[str, float]:
        """One pass over loader. Returns mean loss and mean metrics, weighted by batch size."""
        self.model.train(train)
        totals = {"loss": 0.0, **{k: 0.0 for k in self.metric_fns}}
        n = 0
        for x, y in loader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with torch.set_grad_enabled(train), torch.autocast(self.device.type, enabled=self.use_amp):
                out = self.model(x)
                loss = self.loss_fn(out, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Loss became {loss.item()}; lower the learning rate or check the data")
            if train:
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                if self.cfg.grad_clip:
                    self.scaler.unscale_(self.optimizer)  # clip the real gradients, not the scaled ones
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            b = y.shape[0]
            n += b
            totals["loss"] += loss.item() * b
            out = out.detach().float()
            for name, fn in self.metric_fns.items():
                totals[name] += fn(out, y) * b
        return {k: v / max(n, 1) for k, v in totals.items()}

    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        """Loss and metrics on a loader, no gradients, eval mode (dropout off, BN uses running stats)."""
        return self._run_epoch(loader, train=False)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> list[dict[str, float]]:
        """Train from self.start_epoch to cfg.epochs (or until early stopping). Returns the history."""
        patience = self.cfg.early_stopping_patience
        for epoch in range(self.start_epoch, self.cfg.epochs):
            t0 = time.perf_counter()
            lrs = {f"lr_{g.get('name', i)}": g["lr"] for i, g in enumerate(self.optimizer.param_groups)}
            train_m = self._run_epoch(train_loader, train=True)
            val_m = self.evaluate(val_loader)
            if self.scheduler is not None:
                self.scheduler.step()

            row = {"epoch": epoch + 1, **{f"train_{k}": v for k, v in train_m.items()},
                   **{f"val_{k}": v for k, v in val_m.items()}, **lrs, "epoch_time_s": time.perf_counter() - t0}
            self.history.append(row)
            if self.cfg.monitor not in row:
                raise KeyError(f"train.monitor={self.cfg.monitor!r} is not a logged metric: {sorted(row)}")

            improved = self.sign * row[self.cfg.monitor] > self.sign * self.best_metric
            if improved:
                self.best_metric, self.best_epoch = row[self.cfg.monitor], epoch + 1
            row["best_" + self.cfg.monitor] = self.best_metric
            if self.tracker is not None:
                self.tracker.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch + 1)
            self._save("last.pt", epoch + 1)
            if improved:
                self._save("best.pt", epoch + 1)
            self._write_history()
            logger.info(
                "epoch %d/%d  train_loss %.3f  val_loss %.3f  %s %.4f%s  (%.0fs)",
                epoch + 1, self.cfg.epochs, row["train_loss"], row["val_loss"], self.cfg.monitor,
                row[self.cfg.monitor], "  *best*" if improved else "", row["epoch_time_s"],
            )
            if patience is not None and epoch + 1 - self.best_epoch >= patience:
                logger.info("Early stopping: no %s improvement for %d epochs", self.cfg.monitor, patience)
                break
        return self.history

    # ------------------------------------------------------------ checkpoints
    def _save(self, name: str, epoch: int) -> None:
        """Everything needed to resume or to rebuild the model for evaluation."""
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,  # number of finished epochs
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "monitor": self.cfg.monitor,
            "history": self.history,
            **self.extra_state,
        }
        tmp = self.ckpt_dir / (name + ".tmp")
        torch.save(state, tmp)
        tmp.replace(self.ckpt_dir / name)  # atomic: a disconnect mid-save never leaves a broken file

    def resume(self, path: str | Path) -> None:
        """Continue a run from a last.pt checkpoint (same config)."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and state["scheduler"] is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.start_epoch = state["epoch"]
        self.best_metric, self.best_epoch = state["best_metric"], state["best_epoch"]
        self.history = state["history"]
        logger.info("Resumed from %s at epoch %d (best %s %.4f)", path, self.start_epoch, state["monitor"],
                    self.best_metric)

    def _write_history(self) -> None:
        """history.csv next to the checkpoints: one row per epoch, for plots and the report."""
        path = self.ckpt_dir / "history.csv"
        keys = list(dict.fromkeys(k for row in self.history for k in row))
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.history)
