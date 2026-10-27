"""Per-channel mean and std of the training images, for input normalization.

Normalization: x_norm = (x / 255 - mean) / std, per channel (R, G, B). Networks train
faster and more stably when inputs are centred near 0 with unit spread. The values are
fitted on the TRAIN part only and then reused unchanged for val and test.
"""

import json
import logging
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path

import numpy as np
from datasets import Dataset
from datasets import Image as HFImage
from PIL import Image

logger = logging.getLogger(__name__)


def compute_channel_stats(ds: Dataset, indices: Sequence[int], size: int) -> tuple[list[float], list[float]]:
    """Mean and std of R, G, B over every pixel of the selected images (values scaled to 0..1).

    Each image is converted to RGB and resized to (size, size) first, so every image
    contributes the same number of pixels, like at training time.

    Uses running sums: mean = S / N and std = sqrt(SS / N - mean^2), where S and SS are
    the per-channel sums of x and x^2 over all N pixels (float64 to avoid rounding drift).

    Returns:
        (mean, std), each a list of 3 floats in R, G, B order.
    """
    raw = ds.select_columns(["image"]).cast_column("image", HFImage(decode=False))
    s = np.zeros(3, dtype=np.float64)
    ss = np.zeros(3, dtype=np.float64)
    n = 0
    for i in indices:
        rec = raw[int(i)]["image"]
        img_bytes = rec["bytes"] or Path(rec["path"]).read_bytes()
        with Image.open(BytesIO(img_bytes)) as img:
            x = np.asarray(img.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float64) / 255.0
        pixels = x.reshape(-1, 3)  # (size*size, 3)
        s += pixels.sum(axis=0)
        ss += (pixels ** 2).sum(axis=0)
        n += pixels.shape[0]
    mean = s / n
    std = np.sqrt(ss / n - mean ** 2)
    logger.info("Channel stats over %d images (%d pixels)", len(indices), n)
    return mean.tolist(), std.tolist()


def save_stats(path: str | Path, mean: list[float], std: list[float], **info: object) -> None:
    """Write mean/std (plus how they were computed) to JSON, for training to load later."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mean": mean, "std": std, **info}, indent=2), encoding="utf-8")


def load_stats(path: str | Path) -> tuple[list[float], list[float]]:
    """Read (mean, std) written by save_stats."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["mean"], payload["std"]
