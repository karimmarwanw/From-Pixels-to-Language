"""Task 1.1 EDA, one function per step: python -m src.cli eda --config configs/eda.yaml --step N

These functions are the CLI's reporting layer: they call the reusable logic in
src/data/ and src/visualization/ and print what they find, so printing is fine here.
All steps use the official TRAIN set only; the test set is never loaded.
"""

import random
from pathlib import Path

from src.data.class_stats import class_counts, imbalance_summary
from src.data.loading import get_label_names, load_cub_split
from src.utils.config import EDAConfig
from src.visualization.eda import plot_class_distribution, plot_records


def _header(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def step1_inspect(cfg: EDAConfig) -> None:
    """Load the official train set, print its schema and a few full records, and show their images."""
    train = load_cub_split(cfg.train_name, "train", cfg.cache_dir)
    names = get_label_names(train)

    _header("STEP 1 - Load & inspect (official train set)")
    print(f"Rows:    {len(train)}")
    print(f"Columns: {train.column_names}")
    print("\nSchema:")
    for col, feature in train.features.items():
        kind = type(feature).__name__
        detail = f"{feature.num_classes} classes" if kind == "ClassLabel" else str(feature)
        print(f"  {col:<12} {kind:<11} {detail}")
    print(f"\nClass names: {len(names)}  (first 3: {names[:3]}, last: {names[-1]!r})")

    idx = sorted(random.Random(cfg.seed).sample(range(len(train)), cfg.n_inspect))
    images, titles = [], []
    for i in idx:
        rec = train[i]
        img = rec["image"]
        _header(f"Record {i}")
        print(f"id:          {rec['id']}")
        print(f"file_name:   {rec['file_name']}")
        print(f"label:       {rec['label']} -> {names[rec['label']]}")
        print(f"image:       {img.width}x{img.height} px (W x H), mode {img.mode}, format {img.format}")
        print(f"description: {len(rec['description'])} characters, {rec['description'].count(chr(10))} newline(s)")
        print("raw (repr, so the \\n separators are visible):")
        print(f"  {rec['description']!r}")
        images.append(img)
        titles.append(f"{names[rec['label']]} (label {rec['label']})\n{img.width}x{img.height}, {img.mode}\nrow {i}")

    out = plot_records(images, titles, "Step 1 - three training records",
                       Path(cfg.figures_dir) / "step01_records.png")
    _header("Figure")
    print(f"  {out}")


def step2_class_distribution(cfg: EDAConfig) -> None:
    """Images per class in the official train set: summary, extremes, and two plots."""
    train = load_cub_split(cfg.train_name, "train", cfg.cache_dir)
    counts = class_counts(train["label"], get_label_names(train))  # label column only, no images decoded
    s = imbalance_summary(counts)

    _header("STEP 2 - Class distribution (official train set)")
    print(f"Classes: {s['classes']}   Images: {s['images']}")
    print(f"Images per class -> min {s['min']} | max {s['max']} | mean {s['mean']:.2f} | "
          f"median {s['median']:.0f}")
    print(f"Imbalance ratio (max / min): {s['imbalance_ratio']:.3f}")

    _header("How many classes have N images")
    for n, k in counts["count"].value_counts().sort_index().items():
        print(f"  {n} images: {k:>3} classes")

    smallest = counts[counts["count"] == s["min"]]
    smallest = smallest if len(smallest) > 5 else counts.tail(5)  # show every class tied at the minimum
    _header(f"{len(smallest)} smallest classes")
    print(smallest.sort_values("label").to_string(index=False))
    _header("5 largest classes")
    n_max = (counts["count"] == s["max"]).sum()
    print(f"({n_max} classes are tied at the maximum of {s['max']}; showing the first 5 by label)")
    print(counts.head(5).to_string(index=False))

    out = plot_class_distribution(counts, Path(cfg.figures_dir) / "step02_class_distribution.png")
    _header("Figure")
    print(f"  {out}")


STEPS = {
    1: step1_inspect,
    2: step2_class_distribution,
}
