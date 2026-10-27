"""EDA figures. Each function takes prepared data, saves one PNG, and returns its path."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, no window (works on Kaggle/servers too)
import matplotlib.pyplot as plt
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
