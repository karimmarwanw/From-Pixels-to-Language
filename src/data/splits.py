"""Stratified train/validation split of the official CUB train set.

The official test set is never touched here: validation is carved out of train
so that tuning, early stopping and model selection never see test data.
The indices are saved to JSON and committed so the whole team uses the same split.
"""

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def make_stratified_split(
    labels: Sequence[int],
    val_per_class: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Split row indices into train/val with the same number of val images per class.

    For every class, its indices are shuffled with a seeded RNG and the first
    val_per_class go to val; the rest stay in train. A fixed count (rather than
    a fraction) gives a perfectly balanced val set, since CUB has 29-30 train
    images per class.

    Args:
        labels: class label of each row of the official train set.
        val_per_class: number of images of each class to put in val, e.g. 5.
        seed: RNG seed, so the split is identical on every machine.

    Returns:
        (train_indices, val_indices), both sorted, disjoint, covering all rows.
    """
    if val_per_class < 1:
        raise ValueError(f"val_per_class must be >= 1, got {val_per_class}")

    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []

    for cls in np.unique(labels):
        cls_idx = np.flatnonzero(labels == cls)
        if len(cls_idx) <= val_per_class:
            raise ValueError(
                f"class {cls} has {len(cls_idx)} images, not enough for {val_per_class} val + 1 train"
            )
        rng.shuffle(cls_idx)
        val_idx.extend(cls_idx[:val_per_class].tolist())
        train_idx.extend(cls_idx[val_per_class:].tolist())

    logger.info("Split: %d train, %d val (val_per_class=%d, seed=%d)",
                len(train_idx), len(val_idx), val_per_class, seed)
    return sorted(train_idx), sorted(val_idx)


def save_split(
    path: str | Path,
    train_idx: list[int],
    val_idx: list[int],
    val_per_class: int,
    seed: int,
) -> None:
    """Write the split (plus the settings that produced it) to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "val_per_class": val_per_class,
        "seed": seed,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "train_indices": train_idx,
        "val_indices": val_idx,
    }
    path.write_text(json.dumps(payload))
    logger.info("Saved split to %s", path)


def load_split(path: str | Path) -> tuple[list[int], list[int]]:
    """Read (train_indices, val_indices) saved by save_split."""
    payload = json.loads(Path(path).read_text())
    return payload["train_indices"], payload["val_indices"]
