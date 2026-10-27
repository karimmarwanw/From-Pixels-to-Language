"""Class balance of a split: how many images each class has."""

from collections.abc import Sequence

import pandas as pd


def class_counts(labels: Sequence[int], label_names: list[str]) -> pd.DataFrame:
    """Images per class.

    Args:
        labels: label id of every image.
        label_names: species name of each label id.

    Returns:
        One row per class (all classes, including any with 0 images), columns
        label, class_name, count; sorted by count descending, then label.
    """
    counts = pd.Series(labels).value_counts().reindex(range(len(label_names)), fill_value=0)
    df = pd.DataFrame({"label": counts.index, "class_name": label_names, "count": counts.to_numpy()})
    return df.sort_values(["count", "label"], ascending=[False, True]).reset_index(drop=True)


def imbalance_summary(counts: pd.DataFrame) -> dict[str, float]:
    """min / max / mean / median images per class, and the imbalance ratio max / min."""
    c = counts["count"]
    return {
        "classes": len(c),
        "images": int(c.sum()),
        "min": int(c.min()),
        "max": int(c.max()),
        "mean": float(c.mean()),
        "median": float(c.median()),
        "imbalance_ratio": float(c.max() / c.min()) if c.min() > 0 else float("inf"),
    }
