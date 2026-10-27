"""Stratified train/validation split of the official CUB train set.

Validation is carved out of train so tuning, early stopping and model selection never
need the test set. Duplicate images are kept on the same side of the split, otherwise
val would contain images the model already trained on. The indices are saved to JSON
and committed so the whole team uses the same split.
"""

import json
import logging
from collections.abc import Sequence
from pathlib import Path

from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def make_units(n_rows: int, groups: Sequence[Sequence[int]]) -> list[list[int]]:
    """Units that must not be separated: each duplicate group is one unit, every other row is its own unit.

    Overlapping groups (e.g. a-b and b-c) are merged, so a, b and c end up in one unit.
    """
    parent = list(range(n_rows))  # union-find: each row points to its group's representative

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for group in groups:
        for other in group[1:]:
            parent[find(other)] = find(group[0])

    units: dict[int, list[int]] = {}
    for i in range(n_rows):
        units.setdefault(find(i), []).append(i)
    return list(units.values())


def make_split(
    labels: Sequence[int], val_size: float, seed: int, groups: Sequence[Sequence[int]] = ()
) -> tuple[list[int], list[int]]:
    """Stratified split of row indices into (train, val), keeping each group together.

    Args:
        labels: class label of every row.
        val_size: share of units that go to val, e.g. 0.15.
        seed: random_state, so the split is identical on every machine.
        groups: rows that must stay on the same side (e.g. duplicate pairs). All rows of a
            group must have the same label (true for our duplicates).

    Returns:
        (train_indices, val_indices), sorted, disjoint, together covering all rows.
    """
    units = make_units(len(labels), groups)
    unit_labels = []
    for unit in units:
        unit_classes = {labels[i] for i in unit}
        if len(unit_classes) > 1:
            raise ValueError(f"rows {unit} are grouped together but have different labels {unit_classes}")
        unit_labels.append(unit_classes.pop())

    train_units, val_units = train_test_split(
        units, test_size=val_size, stratify=unit_labels, random_state=seed
    )
    train_idx = sorted(i for unit in train_units for i in unit)
    val_idx = sorted(i for unit in val_units for i in unit)
    logger.info("Split: %d train, %d val (val_size=%.2f, seed=%d, %d grouped units)",
                len(train_idx), len(val_idx), val_size, seed, sum(len(u) > 1 for u in units))
    return train_idx, val_idx


def save_split(
    path: str | Path,
    train_idx: list[int],
    val_idx: list[int],
    file_names: Sequence[str],
    val_size: float,
    seed: int,
    groups: Sequence[Sequence[int]] = (),
) -> None:
    """Write the split to JSON: settings, kept-together groups, indices, and file names (human cross-check)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "official CUB train set (Multimodal-Fatima/CUB_train)",
        "method": "sklearn train_test_split, stratified by label, duplicate groups kept together",
        "val_size": val_size,
        "seed": seed,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "kept_together": [list(g) for g in groups],
        "train_indices": train_idx,
        "val_indices": val_idx,
        "train_files": [file_names[i] for i in train_idx],
        "val_files": [file_names[i] for i in val_idx],
    }
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    logger.info("Saved split to %s", path)


def load_split(path: str | Path) -> tuple[list[int], list[int]]:
    """Read (train_indices, val_indices) written by save_split."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["train_indices"], payload["val_indices"]
