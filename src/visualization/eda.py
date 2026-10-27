"""EDA figures. Each function takes prepared data, saves one PNG, and returns its path."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, no window (works on Kaggle/servers too)
import matplotlib.pyplot as plt
from PIL.Image import Image

# Shared look for all EDA figures: quiet axes and grid, dark text.
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
