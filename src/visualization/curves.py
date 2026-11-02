"""Training curves from the history.csv files the Trainer writes (one row per epoch).

Two kinds of figures:
    - one per run: train vs val loss, and train vs val top-1 accuracy, with the best epoch marked;
    - one comparison of several runs: val top-1 per epoch, and the train - val gap (overfitting).
Train values are measured on augmented images with dropout on, val values on clean images, so
part of the gap comes from that difference and not only from overfitting.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from src.visualization.eda import COLOR_1, COLOR_2, INK, INK_SECONDARY, _save  # same look as the EDA figures

# Categorical lines in a fixed order (validated together for color-vision deficiencies). Slots 3 and 4
# have low contrast on white, so every line also gets a direct text label and its own marker.
SERIES = [(COLOR_1, "o"), (COLOR_2, "s"), ("#1baf7a", "^"), ("#eda100", "D")]


def run_label(history_path: str | Path) -> str:
    """outputs/checkpoints/resnet50-stage2-final_20261029-192701/history.csv -> resnet50-stage2-final."""
    return Path(history_path).parent.name.rsplit("_", 1)[0]


def load_history(path: str | Path) -> pd.DataFrame:
    """One row per epoch: epoch, train_loss, train_acc, val_loss, val_acc, ... (accuracies in [0, 1])."""
    return pd.read_csv(path)


def best_epoch(history: pd.DataFrame) -> tuple[int, float]:
    """(epoch, val_acc) of the best val accuracy; the first one on ties, as the Trainer picks it."""
    i = history["val_acc"].idxmax()
    return int(history.loc[i, "epoch"]), float(history.loc[i, "val_acc"])


def plot_run_curves(history: pd.DataFrame, label: str, out_path: str | Path) -> Path:
    """Left: train vs val loss. Right: train vs val top-1 accuracy (%), best val epoch marked."""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.2))
    e = history["epoch"]
    for ax, col, scale in ((ax_loss, "loss", 1), (ax_acc, "acc", 100)):
        ax.plot(e, history[f"train_{col}"] * scale, color=COLOR_1, lw=2, marker="o", ms=4, label="Train")
        ax.plot(e, history[f"val_{col}"] * scale, color=COLOR_2, lw=2, marker="s", ms=4, label="Val")
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y")
    ax_loss.set_ylabel("Loss (cross-entropy)")
    ax_acc.set_ylabel("Top-1 accuracy (%)")
    be, bv = best_epoch(history)
    ax_acc.scatter([be], [bv * 100], s=70, color=COLOR_2, edgecolors=INK, linewidths=1.2, zorder=4)
    ax_acc.annotate(f"best val {bv:.1%} (epoch {be})", (be, bv * 100), xytext=(0, 12),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=INK,
                    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none"})  # stays readable over lines
    ax_acc.legend(loc="lower right")
    fig.suptitle(f"Training curves: {label}", x=0.01, ha="left", fontsize=12, color=INK)
    fig.text(0.01, -0.02, "Train: augmented images, dropout on. Val: clean images (resize + center crop), "
             "dropout off.", fontsize=8.5, color=INK_SECONDARY)
    fig.tight_layout()
    return _save(fig, out_path)


def _spread(ys: list[float], min_gap: float) -> list[float]:
    """Move label positions apart (keeping their order) so no two are closer than min_gap."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        out[b] = max(out[b], out[a] + min_gap)
    return out


def plot_comparison(histories: dict[str, pd.DataFrame], out_path: str | Path) -> Path:
    """Left: val top-1 per epoch for every run. Right: train - val top-1 gap per epoch (points)."""
    if len(histories) > len(SERIES):
        raise ValueError(f"At most {len(SERIES)} runs per comparison figure, got {len(histories)}")
    fig, (ax_val, ax_gap) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ends: list[tuple[str, float, float, str]] = []
    for (label, h), (color, marker) in zip(histories.items(), SERIES):
        be, bv = best_epoch(h)
        val, gap = h["val_acc"] * 100, (h["train_acc"] - h["val_acc"]) * 100
        ax_val.plot(h["epoch"], val, color=color, lw=2, marker=marker, ms=4,
                    label=f"{label}: best {bv:.1%} (epoch {be})")
        ax_gap.plot(h["epoch"], gap, color=color, lw=2, marker=marker, ms=4)
        ends.append((label, float(h["epoch"].iloc[-1]), float(val.iloc[-1]), color))

    # Direct labels at the line ends, nudged apart so they never overlap
    lo, hi = ax_val.get_ylim()
    ys = _spread([y for _, _, y, _ in ends], min_gap=(hi - lo) * 0.045)
    for (label, x, _, _), y in zip(ends, ys):
        ax_val.annotate(label, (x, y), xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=INK, annotation_clip=False)

    ax_val.set_ylabel("Val top-1 accuracy (%)")
    ax_gap.set_ylabel("Train − val top-1 (points)")
    ax_val.set_title("Val accuracy per epoch", loc="left")
    ax_gap.set_title("Train − val gap (bigger = more overfitting)", loc="left")
    for ax in (ax_val, ax_gap):
        ax.set_xlabel("Epoch of that run")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y")
    ax_gap.axhline(0, color=INK_SECONDARY, lw=0.8)
    fig.legend(*ax_val.get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1),
               fontsize=9)
    fig.suptitle("Scratch CNN vs transfer learning (ResNet50)", x=0.01, ha="left", fontsize=12, color=INK)
    fig.text(0.01, -0.16, "Stage 2 runs start from the stage 1 best checkpoint (epoch 8), so their epoch 1 already "
             "starts high. Train values use augmented images with dropout on.", fontsize=8.5, color=INK_SECONDARY)
    fig.tight_layout()
    return _save(fig, out_path)


def comparison_table(histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The same comparison as a table: epochs, best val top-1/top-5, epoch, train top-1 and gap there."""
    rows = []
    for label, h in histories.items():
        i = h["val_acc"].idxmax()
        rows.append({"run": label, "epochs": len(h), "best_epoch": int(h.loc[i, "epoch"]),
                     "best_val_acc": h.loc[i, "val_acc"], "val_top5": h.loc[i, "val_top5"],
                     "train_acc": h.loc[i, "train_acc"], "gap": h.loc[i, "train_acc"] - h.loc[i, "val_acc"]})
    return pd.DataFrame(rows)
