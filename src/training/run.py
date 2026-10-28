"""Wire config -> data -> model -> optimizer -> Trainer -> W&B for one classification run.

`train_classification(cfg)` is what `cli train` calls. Tuning calls the same function once per
trial with group="<search name>", so every trial is trained and logged exactly like a normal run
and the trials of one search sit together on the W&B dashboard.
"""

import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset as HFDataset

from src.data.datasets import build_classification_loaders
from src.models.classifier import Classifier
from src.models.registry import build_classifier
from src.training.classification import CLASSIFICATION_METRICS, classification_loss
from src.training.optim import build_optimizer, build_scheduler
from src.training.tracking import WandbTracker, environment_tags
from src.training.trainer import Trainer
from src.utils.config import TrainConfig, config_from_dict, flatten
from src.utils.device import get_device
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def load_weights(model: torch.nn.Module, path: str | Path) -> None:
    """Load only the model weights from a checkpoint (e.g. stage 1 head-only -> stage 2 fine-tuning)."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    logger.info("Loaded model weights from %s", path)


def load_classifier(path: str | Path, device: torch.device) -> tuple[Classifier, TrainConfig, list[str]]:
    """Rebuild a trained classifier from a checkpoint alone (its config and class names are stored inside)."""
    state = torch.load(path, map_location=device, weights_only=False)
    cfg = config_from_dict(state["config"])
    class_names = state["class_names"]
    model = build_classifier(cfg.model, num_classes=len(class_names))
    model.load_state_dict(state["model"])
    return model.to(device).eval(), cfg, class_names


def train_classification(
    cfg: TrainConfig, resume: str | Path | None = None, group: str | None = None, hf_train: HFDataset | None = None
) -> dict:
    """Train one classifier and track it in W&B.

    Args:
        resume: path to a last.pt to continue an interrupted run (same W&B run, same folder).
        group: W&B group for tuning trials (one name per search); None = the experiment name.
        hf_train: pre-loaded train split (tests, tuning loops); loaded from the Hub if None.

    Returns:
        dict with best_metric, best_epoch, epochs_run, run_id, ckpt_dir, train_time_s, history.
    """
    set_seed(cfg.seed, cfg.train.deterministic)
    device = get_device(cfg.train.device)
    train_loader, val_loader, class_names = build_classification_loaders(
        cfg.data, cfg.seed, pin_memory=device.type == "cuda", hf_train=hf_train
    )

    model = build_classifier(cfg.model, num_classes=len(class_names))
    if cfg.model.init_from:
        load_weights(model, cfg.model.init_from)  # requires_grad flags from freeze_encoder are kept
    optimizer = build_optimizer(model, cfg.optim)
    scheduler = build_scheduler(optimizer, cfg.optim, cfg.train.epochs)

    run_id = None
    if resume:
        ckpt_dir = Path(resume).parent
        run_id = torch.load(resume, map_location="cpu", weights_only=False).get("wandb_run_id")
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        ckpt_dir = Path(cfg.train.checkpoint_dir) / f"{cfg.run_name}_{stamp}"

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    t = cfg.tracking
    with WandbTracker(t.wandb_project, t.wandb_entity, cfg.experiment, cfg.run_name,
                      run_id=run_id, group=group, mode=t.wandb_mode) as tracker:
        if not resume:
            tracker.log_params(flatten(asdict(cfg)))
            tracker.log_params({
                "n_params": n_params, "n_trainable_params": n_trainable,
                "frozen_stages": sorted(model.encoder._frozen), "n_train": len(train_loader.dataset),
                "n_val": len(val_loader.dataset), "device": device.type,
            })
            tracker.set_tags({**environment_tags(), "checkpoint_dir": str(ckpt_dir)})
        logger.info("Model %s: %.2fM params (%.2fM trainable)", cfg.model.name, n_params / 1e6, n_trainable / 1e6)

        trainer = Trainer(
            model, optimizer, classification_loss(cfg.optim.label_smoothing), device, cfg.train, ckpt_dir,
            scheduler=scheduler, metric_fns=CLASSIFICATION_METRICS, tracker=tracker,
            extra_state={"config": asdict(cfg), "class_names": class_names, "wandb_run_id": tracker.run_id},
        )
        if resume:
            trainer.resume(resume)
        history = trainer.fit(train_loader, val_loader)

        train_time = sum(r["epoch_time_s"] for r in history)
        tracker.log_summary({f"best_{cfg.train.monitor}": trainer.best_metric, "best_epoch": trainer.best_epoch,
                             "epochs_run": len(history), "train_time_s": train_time})
        tracker.log_artifact(ckpt_dir / "history.csv", "history")
        if cfg.tracking.log_checkpoints:
            tracker.log_artifact(ckpt_dir / "best.pt", "model")

    logger.info("Done. Best %s %.4f at epoch %d. Checkpoints in %s", cfg.train.monitor, trainer.best_metric,
                trainer.best_epoch, ckpt_dir)
    return {
        "best_metric": trainer.best_metric, "best_epoch": trainer.best_epoch, "epochs_run": len(history),
        "run_id": tracker.run_id, "ckpt_dir": str(ckpt_dir), "train_time_s": train_time, "history": history,
    }
