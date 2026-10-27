"""Load CUB-200-2011 with captions from the Hugging Face Hub.

Each record has: image (PIL), label (int, 200 classes), description
(10 captions in one string separated by "\\n"), file_name, id.
"""

import logging
from pathlib import Path

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


def load_cub_split(dataset_name: str, split: str, cache_dir: str | Path | None = None) -> Dataset:
    """Load one split, e.g. ("Multimodal-Fatima/CUB_train", "train").

    The first call downloads it into cache_dir; later calls read from disk.
    Images keep their stored mode (no .convert("RGB") here) so the EDA can
    see and count non-RGB images.
    """
    ds = load_dataset(dataset_name, split=split, cache_dir=str(cache_dir) if cache_dir else None)
    logger.info("Loaded %s [%s]: %d rows", dataset_name, split, len(ds))
    return ds


def get_label_names(ds: Dataset) -> list[str]:
    """Species names indexed by label id, read from the dataset features."""
    return ds.features["label"].names
