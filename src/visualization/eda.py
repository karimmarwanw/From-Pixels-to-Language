"""EDA figures. Each function takes prepared data, saves one PNG, and returns its path."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, no window (works on Kaggle/servers too)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL.Image import Image

# Shared look for all EDA figures: one main series color, quiet axes and grid, dark text.
COLOR_1 = "#2a78d6"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.axisbelow": True,  # grid behind the bars
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "legend.frameon": False,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def _save(fig: plt.Figure, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_records(images: list[Image], titles: list[str], suptitle: str, out_path: str | Path) -> Path:
    """Show a few images side by side, each with a multi-line title."""
    fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 4.8))
    for ax, img, title in zip(list(axes) if len(images) > 1 else [axes], images, titles):
        ax.imshow(img, cmap="gray" if img.mode == "L" else None)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, out_path)


def plot_class_distribution(counts: pd.DataFrame, out_path: str | Path) -> Path:
    """Images per class: all classes sorted (left) and a histogram of the counts (right).

    Args:
        counts: output of class_counts() (columns label, class_name, count), sorted by count.
    """
    fig, (ax_bar, ax_hist) = plt.subplots(1, 2, figsize=(16, 4.8), gridspec_kw={"width_ratios": [3, 1]})

    ax_bar.bar(range(len(counts)), counts["count"], width=0.75, color=COLOR_1)
    ax_bar.set_xlim(-1, len(counts))
    ax_bar.set_ylim(0, counts["count"].max() * 1.15)
    ax_bar.set_xlabel(f"Classes, sorted by image count (all {len(counts)})")
    ax_bar.set_ylabel("Images per class")
    ax_bar.set_title("Images per class, sorted", loc="left")
    ax_bar.grid(axis="y")
    n_min = (counts["count"] == counts["count"].min()).sum()
    ax_bar.annotate(
        f"{n_min} classes with {counts['count'].min()} images",
        xy=(len(counts) - 1, counts["count"].min()), xytext=(-10, 22), textcoords="offset points",
        ha="right", fontsize=9, color=INK_SECONDARY,
        arrowprops={"arrowstyle": "-", "color": INK_SECONDARY, "lw": 0.8},
    )

    hist = counts["count"].value_counts().sort_index()
    ax_hist.bar(range(len(hist)), hist.to_numpy(), width=0.6, color=COLOR_1)
    ax_hist.set_xticks(range(len(hist)), [str(k) for k in hist.index])
    for i, n in enumerate(hist.to_numpy()):
        ax_hist.text(i, n, str(n), ha="center", va="bottom", fontsize=10, color=INK_SECONDARY)
    ax_hist.set_ylim(0, hist.max() * 1.12)
    ax_hist.set_xlabel("Images in the class")
    ax_hist.set_ylabel("Number of classes")
    ax_hist.set_title("How many classes have N images", loc="left")
    ax_hist.grid(axis="y")

    fig.suptitle("Step 2 - class distribution of the official train set",
                 fontsize=13, fontweight="bold", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_image_sizes(df: pd.DataFrame, input_size: int, out_path: str | Path) -> Path:
    """Width, height and aspect-ratio histograms, plus a width-vs-height scatter.

    Args:
        df: image stats (columns width, height, aspect).
        input_size: model input side (e.g. 224), drawn as a reference line.
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    ax_w, ax_h, ax_a, ax_s = axes

    for ax, col, label in [(ax_w, "width", "Width (px)"), (ax_h, "height", "Height (px)")]:
        ax.hist(df[col], bins=40, color=COLOR_1)
        ax.axvline(input_size, color=INK_SECONDARY, lw=1.2, ls="--")
        ax.text(input_size, ax.get_ylim()[1] * 0.92, f" model input {input_size}", fontsize=8, color=INK_SECONDARY)
        ax.set_xlabel(label)
        ax.set_ylabel("Images")
        ax.set_title(f"{label.split()[0]}: median {df[col].median():.0f} px", loc="left")
        ax.grid(axis="y")

    ax_a.hist(df["aspect"], bins=50, color=COLOR_1)
    ax_a.axvline(1.0, color=INK_SECONDARY, lw=1.2, ls="--")
    ax_a.text(1.0, ax_a.get_ylim()[1] * 0.92, " square", fontsize=8, color=INK_SECONDARY)
    ax_a.set_xlabel("Aspect ratio (width / height)")
    ax_a.set_ylabel("Images")
    ax_a.set_title(f"Aspect: median {df['aspect'].median():.2f} (landscape)", loc="left")
    ax_a.grid(axis="y")

    ax_s.scatter(df["width"], df["height"], s=6, alpha=0.25, color=COLOR_1, edgecolors="none")
    ax_s.axvline(input_size, color=INK_SECONDARY, lw=1, ls="--")
    ax_s.axhline(input_size, color=INK_SECONDARY, lw=1, ls="--")
    ax_s.set_xlabel("Width (px)")
    ax_s.set_ylabel("Height (px)")
    ax_s.set_title("Width vs height (one dot per image)", loc="left")
    ax_s.grid(True)

    fig.suptitle("Step 3 - image sizes in the official train set",
                 fontsize=13, fontweight="bold", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_color_modes(categories: list[str], counts: list[int], out_path: str | Path) -> Path:
    """Horizontal bars of image color categories on a log axis (so tiny counts stay visible)."""
    fig, ax = plt.subplots(figsize=(9, 2.8))
    y = range(len(categories))
    ax.barh(y, counts, height=0.55, color=COLOR_1)
    for i, n in enumerate(counts):
        ax.text(n * 1.15, i, f"{n:,}", va="center", fontsize=10, color=INK)
    ax.set_yticks(list(y), categories)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(0.8, max(counts) * 4)
    ax.set_xlabel("Images (log scale)")
    ax.set_title("Step 3 - color modes in the official train set", loc="left", fontsize=12, fontweight="bold")
    ax.grid(axis="x")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_image_rows(rows: list[tuple[str, list[Image], list[str]]], suptitle: str, out_path: str | Path) -> Path:
    """A grid with one titled row per group of images, e.g. ('Smallest', [img, ...], [caption, ...])."""
    n_cols = max(len(images) for _, images, _ in rows)
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(3.2 * n_cols, 3.0 * len(rows)), squeeze=False)
    for r, (row_title, images, captions) in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r][c]
            ax.axis("off")
            if c < len(images):
                img = images[c]
                ax.imshow(img, cmap="gray" if img.mode == "L" else None, vmin=0, vmax=255)
                ax.set_title(captions[c], fontsize=8)
        axes[r][0].text(-0.08, 0.5, row_title, transform=axes[r][0].transAxes, rotation=90,
                        ha="right", va="center", fontsize=11, fontweight="bold", color=INK)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_image_pairs(pairs: list[tuple[Image, Image, str, str, str]], suptitle: str, out_path: str | Path) -> Path:
    """One row per pair: left image, right image, and a row label (e.g. the distance).

    Args:
        pairs: (image_a, image_b, caption_a, caption_b, row_label) per pair.
    """
    fig, axes = plt.subplots(len(pairs), 2, figsize=(8, 3.2 * len(pairs)), squeeze=False)
    for r, (img_a, img_b, cap_a, cap_b, row_label) in enumerate(pairs):
        for ax, img, cap in [(axes[r][0], img_a, cap_a), (axes[r][1], img_b, cap_b)]:
            ax.imshow(img, cmap="gray" if img.mode == "L" else None, vmin=0, vmax=255)
            ax.set_title(cap, fontsize=8)
            ax.axis("off")
        axes[r][0].text(-0.06, 0.5, row_label, transform=axes[r][0].transAxes, rotation=90,
                        ha="right", va="center", fontsize=10, fontweight="bold", color=INK)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_nearest_distance(distances: list[int], max_distance: int, out_path: str | Path) -> Path:
    """Histogram of each image's distance to its most similar other image, with the duplicate threshold."""
    fig, ax = plt.subplots(figsize=(10, 4))
    values, counts = np.unique(distances, return_counts=True)
    ax.bar(values, counts, width=1.6, color=COLOR_1)
    ax.axvline(max_distance + 0.9, color=INK_SECONDARY, lw=1.2, ls="--")
    ax.text(max_distance + 1.2, counts.max() * 0.9, f"duplicate if distance <= {max_distance}",
            fontsize=9, color=INK_SECONDARY)
    for v, c in zip(values, counts):
        if v <= max_distance + 2:
            ax.text(v, c, str(c), ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xlabel("pHash Hamming distance to the most similar other image (bits, 0 = identical)")
    ax.set_ylabel("Images")
    ax.set_title("Step 4 - how close is each image to its nearest neighbour?", loc="left",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_image_grid(images: list[Image], titles: list[str], n_cols: int, suptitle: str, out_path: str | Path) -> Path:
    """Images in a grid, each with a (multi-line) title underneath the previous row."""
    n_rows = -(-len(images) // n_cols)  # ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.4 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, img, title in zip(axes.flat, images, titles):
        ax.imshow(img, cmap="gray" if img.mode == "L" else None, vmin=0, vmax=255)
        ax.set_title(title, fontsize=8, loc="left")
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.975))  # leave room so the suptitle does not cover the first row
    return _save(fig, out_path)
