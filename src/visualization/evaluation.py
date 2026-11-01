"""Figures for the classification evaluation. Each function saves one PNG and returns its path."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

from src.evaluation.metrics import species_group
from src.visualization.eda import COLOR_1, COLOR_2, INK_SECONDARY, _save  # same look as the EDA figures


def plot_confused_pairs(pairs: pd.DataFrame, title: str, out_path: str | Path) -> Path:
    """Horizontal bars, largest at the top: how many images of the true species got the wrong label."""
    fig, ax = plt.subplots(figsize=(9, 0.36 * len(pairs) + 1.5))
    y = np.arange(len(pairs))[::-1]  # first row (largest) at the top
    labels = [f"{t}  →  {p}  ({c}/{n})" for t, p, c, n in
              zip(pairs["true"], pairs["predicted"], pairs["count"], pairs["true_total"])]
    ax.barh(y, pairs["count"], height=0.68, color=COLOR_1)
    ax.set_yticks(y, labels, fontsize=8.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Images of the true species predicted as the other species")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    ax.grid(axis="x")
    fig.text(0.0, -0.01, "Label: true species  →  predicted species  (wrong / all images of the true species)",
             fontsize=8.5, color=INK_SECONDARY)
    fig.tight_layout()
    return _save(fig, out_path)


def _group_blocks(class_names: list[str], min_size: int) -> list[tuple[str, int, int]]:
    """(group, first index, last index) for every run of >= min_size neighbouring species of one group."""
    blocks, start = [], 0
    for i in range(1, len(class_names) + 1):
        if i == len(class_names) or species_group(class_names[i]) != species_group(class_names[start]):
            if i - start >= min_size:
                blocks.append((species_group(class_names[start]), start, i - 1))
            start = i
    return blocks


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], title: str, out_path: str | Path,
                          min_group: int = 7) -> Path:
    """Row-normalized 200 x 200 confusion matrix; outlines the large groups (warblers, sparrows, ...)."""
    rates = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)  # each row: where that species' images went
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(rates, cmap="Blues", norm=PowerNorm(gamma=0.5, vmin=0, vmax=1), interpolation="nearest")
    for group, a, b in _group_blocks(class_names, min_group):
        ax.add_patch(Rectangle((a - 0.5, a - 0.5), b - a + 1, b - a + 1, fill=False, ec=COLOR_2, lw=1.2))
        ax.text(b + 2, a - 1, f"{group}s", fontsize=8, color=COLOR_2, va="bottom")
    ticks = np.arange(0, len(class_names), 20)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("Predicted species (class index)")
    ax.set_ylabel("True species (class index)")
    ax.set_title(title, loc="left")
    bar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.08)  # room for the last group's label
    bar.set_label("Share of the true species' images")
    fig.text(0.0, 0.01, "Each row sums to 1. Square-root color scale, so small error rates stay visible. "
             f"Boxes: groups of {min_group}+ related species, which CUB lists next to each other.",
             fontsize=8.5, color=INK_SECONDARY)
    fig.tight_layout()
    return _save(fig, out_path)
