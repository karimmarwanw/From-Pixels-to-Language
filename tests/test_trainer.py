"""Trainer, schedule and end-to-end run tests on tiny synthetic data (CPU, a few seconds)."""

import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.registry import build_classifier
from src.training.classification import CLASSIFICATION_METRICS, classification_loss, topk_accuracy
from src.training.optim import build_optimizer, lr_factor
from src.training.run import load_classifier, train_classification
from src.training.trainer import Trainer
from src.utils.config import ModelConfig, OptimConfig, TrainConfig, TrainingConfig
from tests.test_datasets import fake_cub


def _toy_loaders(n=64, n_classes=4):
    """Class c images have a bright patch in quadrant c, so a CNN can learn it in a few epochs."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, 3, 32, 32, generator=g) * 0.1
    y = torch.arange(n) % n_classes
    for i, c in enumerate(y):
        r, col = divmod(int(c), 2)
        x[i, :, r * 16:(r + 1) * 16, col * 16:(col + 1) * 16] += 2.0
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=16, shuffle=True, generator=g), DataLoader(ds, batch_size=16)


def _toy_trainer(tmp_path, epochs):
    model = build_classifier(ModelConfig(encoder={"channels": [8, 16]}, dropout=0.0), num_classes=4)
    opt = build_optimizer(model, OptimConfig(lr=1e-2, weight_decay=0.0))
    cfg = TrainingConfig(epochs=epochs, amp=False, early_stopping_patience=None)
    return Trainer(model, opt, classification_loss(), torch.device("cpu"), cfg, tmp_path,
                   metric_fns=CLASSIFICATION_METRICS)


def test_topk_accuracy():
    logits = torch.tensor([[0.9, 0.1, 0.0], [0.1, 0.2, 0.7]])
    assert topk_accuracy(1)(logits, torch.tensor([0, 1])) == 0.5
    assert topk_accuracy(2)(logits, torch.tensor([0, 1])) == 1.0


def test_lr_schedule_warmup_then_cosine():
    f = [lr_factor(e, 10, 2, 0.01) for e in range(10)]
    assert f[0] == 0.5 and f[1] == 1.0 and f[2] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(f[2:], f[3:])) and f[-1] > 0.01


def test_backbone_lr_param_groups():
    model = build_classifier(ModelConfig(encoder={"channels": [8]}), num_classes=3)
    opt = build_optimizer(model, OptimConfig(lr=1e-3, backbone_lr=1e-4))
    assert [(g["name"], g["lr"]) for g in opt.param_groups] == [("encoder", 1e-4), ("head", 1e-3)]
    frozen = build_classifier(ModelConfig(encoder={"channels": [8]}, freeze_encoder=True), num_classes=3)
    assert [g["name"] for g in build_optimizer(frozen, OptimConfig()).param_groups] == ["head"]


def test_trainer_learns_and_saves_checkpoints(tmp_path):
    train, val = _toy_loaders()
    trainer = _toy_trainer(tmp_path, epochs=6)
    history = trainer.fit(train, val)
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert trainer.best_metric > 0.5  # 4 classes: chance is 0.25
    assert {"best.pt", "last.pt", "history.csv"} <= {p.name for p in tmp_path.iterdir()}


def test_resume_continues_from_last_epoch(tmp_path):
    train, val = _toy_loaders()
    _toy_trainer(tmp_path, epochs=2).fit(train, val)
    trainer = _toy_trainer(tmp_path, epochs=4)
    trainer.resume(tmp_path / "last.pt")
    assert trainer.start_epoch == 2
    assert [r["epoch"] for r in trainer.fit(train, val)] == [1, 2, 3, 4]


def test_end_to_end_run_with_mlflow(tmp_path, monkeypatch):
    """Full train_classification on a fake CUB split: config -> loaders -> model -> MLflow -> checkpoint."""
    monkeypatch.chdir(tmp_path)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train_indices": list(range(8)), "val_indices": [8, 9, 10, 11]}))
    cfg = TrainConfig(experiment="test", run_name="e2e")
    cfg.data.split_path, cfg.data.batch_size, cfg.data.num_workers, cfg.data.image_size = str(split), 4, 0, 32
    cfg.data.resize_size = 36
    cfg.model.encoder = {"channels": [8, 16]}
    cfg.train.epochs, cfg.train.amp = 2, False
    cfg.tracking.tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    result = train_classification(cfg, hf_train=fake_cub())
    assert result["epochs_run"] == 2 and (tmp_path / "mlflow.db").exists()
    model, loaded_cfg, names = load_classifier(f"{result['ckpt_dir']}/best.pt", torch.device("cpu"))
    assert names == ["species_0", "species_1", "species_2"] and loaded_cfg.run_name == "e2e"
    assert model(torch.randn(1, 3, 32, 32)).shape == (1, 3)
