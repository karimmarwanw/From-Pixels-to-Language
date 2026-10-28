"""PyTorch datasets and DataLoaders for classification, built on Walid's data code.

Pipeline: HF split (loading.py) -> rows chosen by split.json (split.py) -> transform
(transforms.py) -> (image tensor (3, H, W), label int). Batches: (B, 3, H, W) float, (B,) long.

The official test set is only loaded by build_test_loader, which is called only by
`cli evaluate --split test` for final numbers. Training never touches it.
"""

import logging
from collections.abc import Callable, Sequence

import torch
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, Dataset

from src.data.loading import get_label_names, load_cub_split
from src.data.normalization import load_stats
from src.data.split import load_split
from src.data.transforms import build_eval_transform, build_train_transform
from src.utils.config import DataConfig
from src.utils.seed import make_generator, seed_worker

logger = logging.getLogger(__name__)


class CUBClassification(Dataset):
    """Rows `indices` of a Hugging Face CUB split, returned as (transform(image), label)."""

    def __init__(self, hf_ds: HFDataset, indices: Sequence[int], transform: Callable) -> None:
        # Only decode the two columns we need (skips the 10 captions per row).
        self.ds = hf_ds.select_columns(["image", "label"])
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        rec = self.ds[int(self.indices[i])]
        return self.transform(rec["image"]), int(rec["label"])


def get_mean_std(cfg: DataConfig) -> tuple[list[float], list[float]]:
    """Normalization values chosen in the config: ImageNet's, or the CUB train stats (EDA step 8)."""
    if cfg.normalization == "imagenet":
        return cfg.imagenet_mean, cfg.imagenet_std
    if cfg.normalization == "dataset":
        return load_stats(cfg.stats_path)
    raise ValueError(f"normalization must be 'imagenet' or 'dataset', got {cfg.normalization!r}")


def train_transform(cfg: DataConfig) -> Callable:
    mean, std = get_mean_std(cfg)
    return build_train_transform(
        cfg.image_size, mean, std, crop_scale=cfg.crop_scale, hflip_p=cfg.hflip_p,
        rotation_deg=cfg.rotation_deg, brightness=cfg.brightness, contrast=cfg.contrast,
    )


def eval_transform(cfg: DataConfig) -> Callable:
    mean, std = get_mean_std(cfg)
    return build_eval_transform(cfg.image_size, cfg.resize_size, mean, std)


def _loader(ds: Dataset, cfg: DataConfig, shuffle: bool, seed: int, pin: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        drop_last=shuffle,  # train only: a last batch of 1 image breaks BatchNorm
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=cfg.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
    )


def build_classification_loaders(
    cfg: DataConfig, seed: int, pin_memory: bool = False, hf_train: HFDataset | None = None
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Train and val loaders from the official train set and the committed split.

    Args:
        hf_train: an already loaded train split (tests, tuning loops); loaded from the Hub if None.

    Returns:
        (train_loader, val_loader, class_names). class_names[i] is the species of label i.
    """
    hf_train = hf_train if hf_train is not None else load_cub_split(cfg.train_name, "train", cfg.cache_dir)
    train_idx, val_idx = load_split(cfg.split_path)
    if cfg.limit:
        logger.warning("data.limit=%d: using a SUBSET (smoke test only, not for reported results)", cfg.limit)
        train_idx, val_idx = train_idx[: cfg.limit], val_idx[: cfg.limit]
    train_ds = CUBClassification(hf_train, train_idx, train_transform(cfg))
    val_ds = CUBClassification(hf_train, val_idx, eval_transform(cfg))
    logger.info("Train: %d images, val: %d images", len(train_ds), len(val_ds))
    return (
        _loader(train_ds, cfg, shuffle=True, seed=seed, pin=pin_memory),
        _loader(val_ds, cfg, shuffle=False, seed=seed, pin=pin_memory),
        get_label_names(hf_train),
    )


def build_test_loader(cfg: DataConfig, seed: int, pin_memory: bool = False) -> tuple[DataLoader, list[str]]:
    """Loader over the official TEST set. Final evaluation and benchmarking only."""
    hf_test = load_cub_split(cfg.test_name, "test", cfg.cache_dir)
    test_ds = CUBClassification(hf_test, range(len(hf_test)), eval_transform(cfg))
    return _loader(test_ds, cfg, shuffle=False, seed=seed, pin=pin_memory), get_label_names(hf_test)
