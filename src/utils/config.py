"""Load YAML config files into typed dataclasses (EDA config and training configs)."""

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import get_args, get_type_hints

import yaml


@dataclass
class EDAConfig:
    train_name: str
    cache_dir: str
    figures_dir: str
    stats_dir: str
    seed: int = 42
    n_inspect: int = 3
    rebuild_stats: bool = False
    small_side_px: int = 224
    gray_tolerance: int = 3
    n_unusual: int = 5
    wide_aspect: float = 2.0
    tall_aspect: float = 0.5
    dark_brightness: float = 35
    bright_brightness: float = 220
    dup_max_distance: int = 4
    n_lookalikes: int = 4
    n_samples: int = 16
    sample_cols: int = 4
    similar_groups: list[str] = field(default_factory=lambda: ["Gull", "Tern"])
    caption_wrap: int = 40
    n_top_words: int = 30
    min_freq_options: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])
    stop_words: list[str] = field(default_factory=list)
    color_words: list[str] = field(default_factory=list)
    body_part_words: list[str] = field(default_factory=list)
    typo_examples: list[str] = field(default_factory=list)
    n_template_examples: int = 5
    split_path: str = "data/splits/split.json"
    val_size: float = 0.15
    image_size: int = 224
    stats_path: str = "data/processed/stats.json"
    imagenet_mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    imagenet_std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    normalization: str = "imagenet"  # "imagenet" or "dataset" (values from stats_path)
    typo_map_path: str = "configs/caption_typos.yaml"
    min_caption_words: int = 2
    captions_path: str = "data/processed/captions.json"
    resize_size: int = 256
    crop_scale: list[float] = field(default_factory=lambda: [0.5, 1.0])
    hflip_p: float = 0.5
    rotation_deg: float = 15
    brightness: float = 0.2
    contrast: float = 0.2
    aug_example_images: list[int] = field(default_factory=lambda: [204, 5238, 4837])
    eval_example_images: list[int] = field(default_factory=lambda: [204, 5590, 3228, 190])
    n_aug_examples: int = 7
    n_sanity_images: int = 256


def load_eda_config(path: str | Path) -> EDAConfig:
    """Read a YAML file into an EDAConfig. Unknown keys raise an error (catches typos)."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown = set(raw) - {f.name for f in fields(EDAConfig)}
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}")
    return EDAConfig(**raw)


# ---------------------------------------------------------------------------
# Training configs (python -m src.cli train --config configs/cls_*.yaml)
# ---------------------------------------------------------------------------
# One YAML file = one experiment. Sections map 1:1 to the dataclasses below, so every
# setting has a type, a default, and a single place where it is defined.


@dataclass
class DataConfig:
    train_name: str = "Multimodal-Fatima/CUB_train"
    test_name: str = "Multimodal-Fatima/CUB_test"
    cache_dir: str = "data/hf_cache"
    split_path: str = "data/splits/split.json"  # Walid's committed train/val split (EDA step 7)
    image_size: int = 224
    resize_size: int = 256  # eval: resize short side, then center crop image_size
    normalization: str = "imagenet"  # "imagenet" (decided in EDA step 8) or "dataset" (stats_path)
    stats_path: str = "data/processed/stats.json"
    imagenet_mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    imagenet_std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    crop_scale: list[float] = field(default_factory=lambda: [0.5, 1.0])
    hflip_p: float = 0.5
    rotation_deg: float = 15
    brightness: float = 0.2
    contrast: float = 0.2
    batch_size: int = 64
    num_workers: int = 4
    limit: int | None = None  # smoke tests only: keep the first N train and N val rows


@dataclass
class ModelConfig:
    name: str = "scratch_cnn"  # key in src/models/registry.py
    encoder: dict = field(default_factory=dict)  # keyword arguments for the encoder constructor
    dropout: float = 0.3  # before the final linear layer
    freeze_encoder: bool = False  # transfer learning stage 1: train the head only
    init_from: str | None = None  # checkpoint to load model weights from (e.g. stage 1 -> stage 2)


@dataclass
class OptimConfig:
    name: str = "adamw"  # "adamw" or "sgd"
    lr: float = 1e-3  # head learning rate (and encoder, unless backbone_lr is set)
    backbone_lr: float | None = None  # transfer learning: smaller learning rate for the encoder
    weight_decay: float = 5e-4
    momentum: float = 0.9  # sgd only
    label_smoothing: float = 0.0
    scheduler: str = "cosine"  # "cosine" (with linear warmup) or "none"
    warmup_epochs: int = 0
    min_lr_ratio: float = 0.01  # cosine ends at lr * min_lr_ratio


@dataclass
class TrainingConfig:
    epochs: int = 40
    amp: bool = True  # mixed precision, CUDA only
    monitor: str = "val_acc"  # metric used for the best checkpoint and early stopping
    monitor_mode: str = "max"  # "max" or "min"
    early_stopping_patience: int | None = 10  # epochs without improvement; null = off
    grad_clip: float | None = None  # max gradient norm; null = off
    checkpoint_dir: str = "outputs/checkpoints"
    device: str = "auto"
    deterministic: bool = False  # True = deterministic cuDNN (slower)


@dataclass
class TrackingConfig:
    tracking_uri: str = "mlruns"  # local folder; open with: mlflow ui --backend-store-uri ./mlruns
    log_checkpoints: bool = False  # True copies .pt files into MLflow (~100 MB each for ResNet50)


@dataclass
class TrainConfig:
    experiment: str  # MLflow experiment, e.g. cls-scratch, cls-transfer
    run_name: str  # says what changed, e.g. scratch-cnn-baseline
    task: str = "classification"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)


def _build(cls: type, raw: dict, where: str):
    """Recursively turn a dict into dataclass `cls`, rejecting unknown keys and fixing number types.

    PyYAML reads `1e-3` (no dot) as the STRING "1e-3", so float fields are converted explicitly.
    """
    hints = get_type_hints(cls)
    unknown = set(raw) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown keys in {where}: {sorted(unknown)}")
    kwargs = {}
    for name, value in raw.items():
        hint = hints[name]
        if is_dataclass(hint):
            kwargs[name] = _build(hint, value or {}, f"{where}.{name}")
        elif value is not None and float in get_args(hint) + (hint,) and isinstance(value, (int, str)):
            kwargs[name] = float(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def apply_overrides(raw: dict, overrides: list[str]) -> dict:
    """Apply command line overrides like ["optim.lr=3e-4", "train.epochs=5"] to a raw config dict.

    Values are parsed as YAML, so numbers, booleans, null and lists work: data.crop_scale=[0.6,1.0].
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got {item!r}")
        key, value = item.split("=", 1)
        node = raw
        *parents, last = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = yaml.safe_load(value)
    return raw


def load_train_config(path: str | Path, overrides: list[str] | None = None) -> TrainConfig:
    """Read a training YAML (plus optional overrides) into a TrainConfig."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    raw = apply_overrides(raw, overrides or [])
    return _build(TrainConfig, raw, str(path))


def config_from_dict(raw: dict) -> TrainConfig:
    """Rebuild a TrainConfig from asdict() output (e.g. the copy stored inside a checkpoint)."""
    return _build(TrainConfig, raw, "checkpoint config")


def flatten(d: dict, prefix: str = "") -> dict[str, object]:
    """{"optim": {"lr": 0.001}} -> {"optim.lr": 0.001}, for logging params to MLflow."""
    out: dict[str, object] = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            out.update(flatten(value, f"{name}."))
        else:
            out[name] = value
    return out
