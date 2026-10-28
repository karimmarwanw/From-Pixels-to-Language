"""Quick tests for the classification dataset/loaders, on a tiny fake CUB split (no download)."""

import json

import torch
from datasets import ClassLabel, Dataset, Features, Image, Value
from PIL import Image as PILImage

from src.data.datasets import CUBClassification, build_classification_loaders, eval_transform
from src.utils.config import DataConfig


def fake_cub(n: int = 12, n_classes: int = 3) -> Dataset:
    """A Hugging Face dataset shaped like Multimodal-Fatima/CUB_train (one grayscale image included)."""
    images = [PILImage.new("L" if i == 0 else "RGB", (500, 375), 100) for i in range(n)]
    features = Features({
        "image": Image(),
        "label": ClassLabel(names=[f"species_{c}" for c in range(n_classes)]),
        "description": Value("string"),
    })
    return Dataset.from_dict(
        {"image": images, "label": [i % n_classes for i in range(n)], "description": ["a bird.\n"] * n},
        features=features,
    )


def test_dataset_item_shape_and_label():
    ds = CUBClassification(fake_cub(), [0, 4], eval_transform(DataConfig()))
    x, y = ds[1]
    assert x.shape == (3, 224, 224) and x.dtype == torch.float32 and y == 1
    assert ds[0][0].shape == (3, 224, 224)  # grayscale row is converted to RGB


def test_loaders_use_the_split_file(tmp_path):
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train_indices": list(range(9)), "val_indices": [9, 10, 11]}))
    cfg = DataConfig(split_path=str(split), batch_size=4, num_workers=0)
    train, val, names = build_classification_loaders(cfg, seed=0, hf_train=fake_cub())
    xb, yb = next(iter(train))
    assert xb.shape == (4, 3, 224, 224) and yb.dtype == torch.long
    assert len(train) == 2 and len(val.dataset) == 3  # 9 // 4 with drop_last; val keeps all
    assert names == ["species_0", "species_1", "species_2"]
