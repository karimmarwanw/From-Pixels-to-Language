"""Loading CUB-200-2011 with captions from the Hugging Face Hub.

Each record has: image (PIL), label (int, 200 classes), description
(10 captions in one string separated by "\\n"), file_name, id.
"""

import logging
from pathlib import Path

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


def load_cub(
    train_name: str,
    test_name: str,
    cache_dir: str | Path | None = None,
) -> tuple[Dataset, Dataset]:
    """Load the official CUB train and test splits.

    Args:
        train_name: HF dataset id for the train split, e.g. "Multimodal-Fatima/CUB_train".
        test_name: HF dataset id for the test split, e.g. "Multimodal-Fatima/CUB_test".
        cache_dir: where HF stores the download. The first call downloads ~1.1 GB;
            later calls read from this folder instead of re-downloading.

    Returns:
        (train, test) Hugging Face datasets. Images stay as PIL; RGB conversion
        happens later so the EDA can still count non-RGB images.
    """
    cache_dir = str(cache_dir) if cache_dir is not None else None
    train = load_dataset(train_name, split="train", cache_dir=cache_dir)
    test = load_dataset(test_name, split="test", cache_dir=cache_dir)
    logger.info("Loaded CUB: %d train, %d test images", len(train), len(test))
    return train, test


def get_label_names(ds: Dataset) -> list[str]:
    """Species names indexed by label id, read from the dataset features."""
    return ds.features["label"].names
