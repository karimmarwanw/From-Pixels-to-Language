"""Exact and near-duplicate image detection.

- Exact duplicates: identical file bytes -> identical MD5.
- Near-duplicates: perceptual hash (pHash). pHash keeps the low-frequency structure of
  the image in 64 bits, so a resized / recompressed / lightly edited copy gets a hash
  that differs in only a few bits. The number of differing bits is the Hamming distance.

find_near_duplicates() compares two sets, so the same code serves "within train" now
and "train vs test" at final evaluation.
"""

import hashlib
import logging
from io import BytesIO
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from datasets import Dataset
from datasets import Image as HFImage
from PIL import Image

logger = logging.getLogger(__name__)


def compute_hashes(ds: Dataset) -> pd.DataFrame:
    """MD5 of the raw file bytes and 64-bit pHash of the image, for every row.

    Returns:
        Columns index, label, md5, phash (16-character hex string).
    """
    raw = ds.select_columns(["image", "label"]).cast_column("image", HFImage(decode=False))
    rows = []
    for index, rec in enumerate(raw):
        img_bytes = rec["image"]["bytes"] or Path(rec["image"]["path"]).read_bytes()
        with Image.open(BytesIO(img_bytes)) as img:
            phash = str(imagehash.phash(img.convert("RGB")))
        rows.append({"index": index, "label": rec["label"],
                     "md5": hashlib.md5(img_bytes).hexdigest(), "phash": phash})
    logger.info("Hashed %d images", len(rows))
    return pd.DataFrame(rows)


def load_or_build_hashes(ds: Dataset, csv_path: str | Path, rebuild: bool = False) -> pd.DataFrame:
    """Return the cached hash CSV if it exists (and rebuild is False), else compute and write it."""
    csv_path = Path(csv_path)
    if csv_path.exists() and not rebuild:
        logger.info("Reading cached hashes from %s", csv_path)
        return pd.read_csv(csv_path, dtype={"md5": str, "phash": str})
    df = compute_hashes(ds)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def exact_duplicate_groups(hashes: pd.DataFrame) -> list[list[int]]:
    """Groups of row indices whose files are byte-identical (same MD5). Only groups of 2+."""
    groups = hashes.groupby("md5")["index"].apply(list)
    return [g for g in groups if len(g) > 1]


DUPLICATE_TYPES = ["exact", "resized", "edited / re-saved"]


def duplicate_type(md5_a: str, md5_b: str, size_a: tuple[int, int], size_b: tuple[int, int]) -> str:
    """Name the kind of a duplicate pair (already found by pHash).

    exact             = identical file bytes (same MD5)
    resized           = different bytes and a different (width, height)
    edited / re-saved = different bytes but the same size: pixels changed
                        (brightness/contrast edit or JPEG recompression), no resize
    """
    if md5_a == md5_b:
        return "exact"
    if size_a != size_b:
        return "resized"
    return "edited / re-saved"


def _to_uint64(phashes: pd.Series) -> np.ndarray:
    return np.array([int(h, 16) for h in phashes], dtype=np.uint64)


def nearest_distances(hashes_a: pd.Series, hashes_b: pd.Series | None = None, chunk: int = 512) -> np.ndarray:
    """For every hash in A, the Hamming distance to its closest hash in B (or in A itself, excluding itself).

    Returns:
        (len(A),) int array of distances 0..64.
    """
    a = _to_uint64(hashes_a)
    b = a if hashes_b is None else _to_uint64(hashes_b)
    best = np.full(len(a), 64, dtype=np.int64)
    for start in range(0, len(a), chunk):
        d = np.bitwise_count(a[start:start + chunk, None] ^ b[None, :]).astype(np.int64)  # (chunk, len(B))
        if hashes_b is None:  # ignore each image's distance to itself
            rows = np.arange(d.shape[0])
            d[rows, start + rows] = 64
        best[start:start + chunk] = d.min(axis=1)
    return best


def find_near_duplicates(
    hashes_a: pd.Series, max_distance: int, hashes_b: pd.Series | None = None, chunk: int = 512
) -> pd.DataFrame:
    """All pairs whose pHash Hamming distance is <= max_distance.

    Args:
        hashes_a: pHash hex strings of set A (position = row index).
        max_distance: largest number of differing bits that still counts as a near-duplicate.
        hashes_b: set B; None means "compare A with itself" (each pair reported once).

    Returns:
        Columns a, b (row positions in A and B) and distance, sorted by distance.
    """
    a = _to_uint64(hashes_a)
    b = a if hashes_b is None else _to_uint64(hashes_b)
    pairs = []
    for start in range(0, len(a), chunk):
        d = np.bitwise_count(a[start:start + chunk, None] ^ b[None, :])  # (chunk, len(B))
        ia, ib = np.nonzero(d <= max_distance)
        ia = ia + start
        keep = ia < ib if hashes_b is None else np.ones_like(ia, dtype=bool)  # within one set: i < j only
        pairs += [(int(i), int(j), int(d[i - start, j])) for i, j in zip(ia[keep], ib[keep])]
    return pd.DataFrame(pairs, columns=["a", "b", "distance"]).sort_values("distance").reset_index(drop=True)
