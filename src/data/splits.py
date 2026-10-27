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
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Split row indices into train/val, keeping each class's proportion.

    For every class, its indices are shuffled with a seeded RNG and
    round(val_fraction * n_class) of them go to val (at least 1).

    Args:
        labels: class label of each row of the official train set.
        val_fraction: share of each class to put in val, e.g. 0.15.
        seed: RNG seed, so the split is identical on every machine.

    Returns:
        (train_indices, val_indices), both sorted, disjoint, covering all rows.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []

    for cls in np.unique(labels):
        cls_idx = np.flatnonzero(labels == cls)
        rng.shuffle(cls_idx)
        # floor(x + 0.5) instead of round(): Python's round(4.5) == 4 (banker's rounding)
        n_val = max(1, int(np.floor(val_fraction * len(cls_idx) + 0.5)))
        val_idx.extend(cls_idx[:n_val].tolist())
        train_idx.extend(cls_idx[n_val:].tolist())

    logger.info("Split: %d train, %d val (val_fraction=%.2f, seed=%d)",
                len(train_idx), len(val_idx), val_fraction, seed)
    return sorted(train_idx), sorted(val_idx)


def save_split(
    path: str | Path,
    train_idx: list[int],
    val_idx: list[int],
    val_fraction: float,
    seed: int,
) -> None:
    """Write the split (plus the settings that produced it) to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "val_fraction": val_fraction,
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
