"""Task 1.1 EDA, one function per step: python -m src.cli eda --config configs/eda.yaml --step N

These functions are the CLI's reporting layer: they call the reusable logic in
src/data/ and src/visualization/ and print what they find, so printing is fine here.
All steps use the official TRAIN set only; the test set is never loaded.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.class_stats import class_counts, imbalance_summary
from src.data.duplicates import (
    DUPLICATE_TYPES,
    duplicate_type,
    exact_duplicate_groups,
    find_near_duplicates,
    load_or_build_hashes,
    nearest_distances,
)
from src.data.image_stats import load_or_build_image_stats
from src.data.loading import get_label_names, load_cub_split
from src.utils.config import EDAConfig
from src.visualization.eda import (
    plot_class_distribution,
    plot_color_modes,
    plot_image_pairs,
    plot_image_rows,
    plot_image_sizes,
    plot_nearest_distance,
    plot_records,
)


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


def step3_image_properties(cfg: EDAConfig) -> None:
    """Sizes, aspect ratios, color modes, corrupted / small / gray / dark / bright images."""
    train = load_cub_split(cfg.train_name, "train", cfg.cache_dir)
    names = get_label_names(train)
    df = load_or_build_image_stats(
        train, names, Path(cfg.stats_dir) / "image_stats.csv", cfg.gray_tolerance, cfg.rebuild_stats
    )
    ok = df[df["ok"]]
    cols = ["index", "class_name", "width", "height", "mode"]

    _header("STEP 3 - Image properties & quality (official train set)")
    print(f"Images scanned: {len(df)}   decoded OK: {len(ok)}   formats: {df['format'].value_counts().to_dict()}")
    print("\nSummary (px, bytes, brightness 0-255):")
    summary = ok[["width", "height", "aspect", "short_side", "n_bytes", "mean_brightness"]].describe()
    print(summary.loc[["min", "25%", "50%", "mean", "75%", "max"]].round(2).to_string())
    longest_500 = ((ok["width"] == 500) | (ok["height"] == 500)).mean()
    print(f"\nImages whose longest side is 500 px: {longest_500:.1%}")

    _header("Corrupted files (fail to decode)")
    bad = df[~df["ok"]]
    print(f"Count: {len(bad)}")
    if len(bad):
        print(bad[["index", "file_name", "error"]].to_string(index=False))

    _header("Color modes")
    print(df["mode"].value_counts().to_string())
    gray = ok[ok["gray_content"].astype(bool)]
    print(f"\nGrayscale images ({len(gray)}): stored as 'L', or stored as RGB but with no color "
          f"(every pixel has |R-G|,|G-B|,|R-B| <= {cfg.gray_tolerance}):")
    print(gray[cols + ["max_channel_diff"]].to_string(index=False))
    near = ok[(~ok["gray_content"].astype(bool)) & (ok["max_channel_diff"] <= 20)]
    print(f"\nNearly gray (channel difference {cfg.gray_tolerance + 1}-20), kept as color: {len(near)}")
    if len(near):
        print(near[cols + ["max_channel_diff"]].to_string(index=False))

    _header(f"Small images (shorter side < {cfg.small_side_px} px, will be upscaled)")
    small = ok[ok["short_side"] < cfg.small_side_px].sort_values("short_side")
    print(f"Count: {len(small)} ({len(small) / len(ok):.1%})")
    print(small[cols + ["short_side"]].head(10).to_string(index=False))

    wide = ok[ok["aspect"] > cfg.wide_aspect].sort_values("aspect", ascending=False)
    tall = ok[ok["aspect"] < cfg.tall_aspect].sort_values("aspect")
    dark = ok[ok["mean_brightness"] < cfg.dark_brightness].sort_values("mean_brightness")
    bright = ok[ok["mean_brightness"] > cfg.bright_brightness].sort_values("mean_brightness", ascending=False)

    _header(f"Extreme aspect ratios (wide: > {cfg.wide_aspect}, tall: < {cfg.tall_aspect})")
    print(f"Very wide: {len(wide)}");  print(wide[cols + ["aspect"]].head(5).round(2).to_string(index=False))
    print(f"\nVery tall: {len(tall)}"); print(tall[cols + ["aspect"]].head(5).round(2).to_string(index=False))

    _header(f"Very dark (< {cfg.dark_brightness}) and very bright (> {cfg.bright_brightness}); "
            "mean pixel value, 0 = black, 255 = white")
    print(f"Very dark: {len(dark)}");     print(dark[cols + ["mean_brightness"]].head(5).round(1).to_string(index=False))
    print(f"\nVery bright: {len(bright)}"); print(bright[cols + ["mean_brightness"]].head(5).round(1).to_string(index=False))

    categories = [
        ("Corrupted (cannot be opened)", bad, "fails to decode"),
        ("Grayscale", gray, f"mode L, or RGB with channel difference <= {cfg.gray_tolerance}"),
        ("Small", small, f"shorter side < {cfg.small_side_px} px"),
        ("Very wide", wide, f"aspect > {cfg.wide_aspect}"),
        ("Very tall", tall, f"aspect < {cfg.tall_aspect}"),
        ("Very dark", dark, f"mean brightness < {cfg.dark_brightness}"),
        ("Very bright", bright, f"mean brightness > {cfg.bright_brightness}"),
    ]
    _header("Summary: unusual images (official train set)")
    print(f"{'category':<30}{'count':>6}{'share':>8}   rule")
    for name, sub, rule in categories:
        print(f"{name:<30}{len(sub):>6}{len(sub) / len(df):>8.1%}   {rule}")
    flagged = set().union(*(set(sub["index"]) for _, sub, _ in categories))
    print(f"{'Any of the above (unique)':<30}{len(flagged):>6}{len(flagged) / len(df):>8.1%}"
          "   an image can be in several categories")

    # ------------------------------------------------------------------ figures
    figs = Path(cfg.figures_dir)
    out_sizes = plot_image_sizes(ok, cfg.small_side_px, figs / "step03_image_sizes.png")
    n_rgb_gray = int((gray["mode"] == "RGB").sum())
    out_modes = plot_color_modes(
        ["RGB (color)", "RGB, but no color", "L (grayscale)"],
        [int((ok["mode"] == "RGB").sum()) - n_rgb_gray, n_rgb_gray, int((ok["mode"] == "L").sum())],
        figs / "step03_color_modes.png",
    )
    n = cfg.n_unusual
    groups = [
        (f"Grayscale ({len(gray)})", gray.head(n), lambda r: f"{r['mode']}, max diff {r['max_channel_diff']}"),
        (f"Small ({len(small)})", small.head(n), lambda r: f"{r['width']}x{r['height']}"),
        (f"Wide {len(wide)} / tall {len(tall)}", pd.concat([wide.head(n - min(len(tall), 2)), tall.head(2)]),
         lambda r: f"aspect {r['aspect']:.2f}"),
        (f"Very dark ({len(dark)})", dark.head(n), lambda r: f"brightness {r['mean_brightness']:.0f}"),
        (f"Very bright ({len(bright)})", bright.head(n), lambda r: f"brightness {r['mean_brightness']:.0f}"),
    ]
    rows = []
    for title, sub, describe in groups:
        images = [train[int(i)]["image"] for i in sub["index"]]
        captions = [f"{r['class_name']}\nrow {r['index']} | {describe(r)}" for _, r in sub.iterrows()]
        rows.append((title, images, captions))
    out_grid = plot_image_rows(rows, "Step 3 - unusual training images", figs / "step03_unusual_images.png")

    _header("Files")
    for p in [Path(cfg.stats_dir) / "image_stats.csv", out_sizes, out_modes, out_grid]:
        print(f"  {p}")


def step4_duplicates(cfg: EDAConfig) -> None:
    """Exact (MD5) and near (pHash) duplicates within the official train set."""
    train = load_cub_split(cfg.train_name, "train", cfg.cache_dir)
    names = get_label_names(train)
    hashes = load_or_build_hashes(train, Path(cfg.stats_dir) / "image_hashes.csv", cfg.rebuild_stats)

    def describe(i: int) -> str:
        img = train[i]["image"]
        return f"row {i:>4}  {names[hashes.label[i]]:<26} {img.width}x{img.height}"

    _header("STEP 4 - Duplicates (official train set)")
    exact = exact_duplicate_groups(hashes)
    print(f"Exact duplicates (identical file bytes, same MD5): {len(exact)} group(s)")
    for g in exact:
        print("  " + "  =  ".join(describe(i) for i in g))

    near = find_near_duplicates(hashes["phash"], cfg.dup_max_distance)
    near["same_class"] = hashes.label[near.a].to_numpy() == hashes.label[near.b].to_numpy()
    near["type"] = [
        duplicate_type(hashes.md5[a], hashes.md5[b], train[a]["image"].size, train[b]["image"].size)
        for a, b in zip(near.a, near.b)
    ]
    _header(f"All duplicate pairs (pHash Hamming distance <= {cfg.dup_max_distance} of 64 bits)")
    print(f"Pairs: {len(near)}   same class: {near.same_class.sum()}   different class: {(~near.same_class).sum()}")
    for r in near.itertuples():
        print(f"  [{r.type}] d={r.distance}:  {describe(r.a)}   ~   {describe(r.b)}")
    involved = sorted(set(near.a) | set(near.b))
    print(f"\nImages involved: {len(involved)}  ->  extra copies: {len(involved) - len(near)}")

    _header("Duplicate pairs by type")
    print(f"{'type':<20}{'pairs':>6}   meaning")
    meaning = {
        "exact": "identical file bytes (same MD5)",
        "resized": "same photo, different width x height",
        "edited / re-saved": "same photo and size, pixels changed (brightness/contrast or recompression)",
    }
    for t in DUPLICATE_TYPES:
        print(f"{t:<20}{(near.type == t).sum():>6}   {meaning[t]}")
    print(f"{'total':<20}{len(near):>6}")

    nearest = nearest_distances(hashes["phash"])
    _header("Distance from each image to its most similar other image")
    for d, n in zip(*np.unique(nearest, return_counts=True)):
        if d <= cfg.dup_max_distance + 6:
            print(f"  distance {d:>2}: {n:>5} images")
    print(f"  median: {np.median(nearest):.0f} bits  -> a typical image is far from every other image")

    lookalikes = find_near_duplicates(hashes["phash"], cfg.dup_max_distance + 4)
    lookalikes = lookalikes[lookalikes.distance > cfg.dup_max_distance].head(cfg.n_lookalikes)

    # ------------------------------------------------------------------ figures
    def pair(a: int, b: int, label: str) -> tuple:
        ia, ib = train[a]["image"], train[b]["image"]
        return (ia, ib, f"{names[hashes.label[a]]}\nrow {a} | {ia.width}x{ia.height}",
                f"{names[hashes.label[b]]}\nrow {b} | {ib.width}x{ib.height}", label)

    figs = Path(cfg.figures_dir)
    out_dups = plot_image_pairs([pair(r.a, r.b, f"{r.type}, d = {r.distance}") for r in near.itertuples()],
                                "Step 4 - duplicate pairs in the train set", figs / "step04_duplicate_pairs.png")
    out_look = plot_image_pairs([pair(r.a, r.b, f"d = {r.distance}") for r in lookalikes.itertuples()],
                                "Step 4 - closest pairs that are NOT duplicates",
                                figs / "step04_closest_non_duplicates.png")
    out_hist = plot_nearest_distance(nearest.tolist(), cfg.dup_max_distance, figs / "step04_nearest_distance.png")

    _header("Files")
    for p in [Path(cfg.stats_dir) / "image_hashes.csv", out_dups, out_look, out_hist]:
        print(f"  {p}")


STEPS = {
    1: step1_inspect,
    2: step2_class_distribution,
    3: step3_image_properties,
    4: step4_duplicates,
}
