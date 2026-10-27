"""Per-image properties for the EDA: size, color mode, file size, decodability, brightness, gray content.

One pass over the images produces a table with one row per image. It is cached to
CSV because decoding thousands of JPEGs takes a while; later steps reuse it.
"""

import logging
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from datasets import Image as HFImage
from PIL import Image

logger = logging.getLogger(__name__)

THUMB_SIZE = (64, 64)  # brightness / gray checks run on a small copy: fast, and enough for averages


def _pixel_stats(img: Image.Image, gray_tolerance: int) -> tuple[float, float, bool]:
    """Brightness and gray-content of one image, measured on a 64x64 RGB thumbnail.

    Returns:
        mean_brightness: mean of all RGB values, 0 (black) .. 255 (white).
        max_channel_diff: largest |R-G|, |G-B| or |R-B| of any pixel.
        gray_content: True if max_channel_diff <= gray_tolerance, i.e. the image has
            no real color even if it is stored as RGB (small tolerance for JPEG noise).
    """
    rgb = np.asarray(img.convert("RGB").resize(THUMB_SIZE), dtype=np.int16)  # (64, 64, 3)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    max_diff = int(max(np.abs(r - g).max(), np.abs(g - b).max(), np.abs(r - b).max()))
    return float(rgb.mean()), max_diff, max_diff <= gray_tolerance


def collect_image_stats(ds: Dataset, label_names: list[str], gray_tolerance: int) -> pd.DataFrame:
    """Scan every image of a split.

    Images are read as raw bytes (decode=False) so we see each file exactly as stored,
    before any .convert("RGB"). img.load() forces a full decode, which is what catches
    truncated or corrupted JPEGs (opening only reads the header).

    Returns:
        One row per image: index, label, class_name, file_name, width, height, aspect (W/H),
        short_side, mode, format, n_bytes, ok, error, mean_brightness, max_channel_diff, gray_content.
    """
    raw = ds.select_columns(["image", "label", "file_name"]).cast_column("image", HFImage(decode=False))
    rows = []
    for index, rec in enumerate(raw):
        img_bytes = rec["image"]["bytes"] or Path(rec["image"]["path"]).read_bytes()
        row = {
            "index": index, "label": rec["label"], "class_name": label_names[rec["label"]],
            "file_name": rec["file_name"], "n_bytes": len(img_bytes),
            "width": None, "height": None, "mode": None, "format": None, "ok": False, "error": "",
            "mean_brightness": None, "max_channel_diff": None, "gray_content": None,
        }
        try:
            with Image.open(BytesIO(img_bytes)) as img:
                row.update(width=img.width, height=img.height, mode=img.mode, format=img.format)
                img.load()
                brightness, max_diff, is_gray = _pixel_stats(img, gray_tolerance)
                row.update(mean_brightness=brightness, max_channel_diff=max_diff, gray_content=is_gray)
            row["ok"] = True
        except Exception as exc:  # any decode failure = corrupted file: recorded, not raised
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    df = pd.DataFrame(rows)
    df["aspect"] = df["width"] / df["height"]
    df["short_side"] = df[["width", "height"]].min(axis=1)
    logger.info("Scanned %d images (%d failed to decode)", len(df), (~df["ok"]).sum())
    return df


def load_or_build_image_stats(
    ds: Dataset, label_names: list[str], csv_path: str | Path, gray_tolerance: int, rebuild: bool = False
) -> pd.DataFrame:
    """Return the cached CSV if it exists (and rebuild is False), else scan and write it."""
    csv_path = Path(csv_path)
    if csv_path.exists() and not rebuild:
        logger.info("Reading cached image stats from %s", csv_path)
        return pd.read_csv(csv_path, keep_default_na=False, na_values=[""])
    df = collect_image_stats(ds, label_names, gray_tolerance)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info("Wrote image stats to %s", csv_path)
    return df
