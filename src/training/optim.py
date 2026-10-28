"""Optimizer and learning-rate schedule from config.

Two parameter groups: encoder and head. With optim.backbone_lr set (transfer learning), the
pretrained encoder gets a smaller learning rate than the new random head, so fine-tuning does
not wreck the pretrained features. Frozen parameters (requires_grad=False) are left out.
"""

import math

import torch
from torch import nn

from src.utils.config import OptimConfig


def build_optimizer(model: nn.Module, cfg: OptimConfig) -> torch.optim.Optimizer:
    """Optimizer over model.encoder (lr = backbone_lr or lr) and everything else (lr)."""
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    enc_ids = {id(p) for p in model.encoder.parameters()}
    head_params = [p for p in model.parameters() if p.requires_grad and id(p) not in enc_ids]
    groups = []
    if enc_params:
        groups.append({"params": enc_params, "lr": cfg.backbone_lr or cfg.lr, "name": "encoder"})
    groups.append({"params": head_params, "lr": cfg.lr, "name": "head"})

    if cfg.name == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.name == "sgd":
        return torch.optim.SGD(groups, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay, nesterov=True)
    raise ValueError(f"optim.name must be 'adamw' or 'sgd', got {cfg.name!r}")


def lr_factor(epoch: int, epochs: int, warmup_epochs: int, min_ratio: float) -> float:
    """Multiplier of the base lr at `epoch` (0-based): linear warmup, then cosine decay to min_ratio.

    Warmup: epoch e < W gives (e + 1) / W. Afterwards: min_ratio + (1 - min_ratio) * 0.5 * (1 + cos(pi * t)),
    where t goes from 0 to 1 over the remaining epochs. Big steps early, small careful steps at the end.
    """
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    t = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: OptimConfig, epochs: int):
    """Per-epoch scheduler (call .step() once at the end of every epoch), or None."""
    if cfg.scheduler == "none":
        return None
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda e: lr_factor(e, epochs, cfg.warmup_epochs, cfg.min_lr_ratio)
        )
    raise ValueError(f"optim.scheduler must be 'cosine' or 'none', got {cfg.scheduler!r}")
