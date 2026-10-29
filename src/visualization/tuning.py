"""Figures for the grid vs random search comparison (Task 1.2). Each function saves one PNG and returns its path.

The metric is an accuracy in [0, 1] (best_val_acc); the figures show it in percent.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from src.tuning.compare import SearchResults, best_so_far
from src.visualization.eda import COLOR_1, COLOR_2, INK, INK_SECONDARY, _save  # same look as the EDA figures

# Each method keeps the same color AND marker in every figure, so the two can be told apart
# without color too (grayscale print, color blindness).
METHOD_STYLE = {
    "grid": {"color": COLOR_1, "marker": "o", "label": "Grid search"},
    "random": {"color": COLOR_2, "marker": "^", "label": "Random search"},
}
DOT = {"s": 42, "edgecolors": "white", "linewidths": 1.0, "zorder": 3}  # white ring keeps overlapping dots visible


def _legend_handle(method: str, label: str) -> Line2D:
    st = METHOD_STYLE[method]
    return Line2D([0], [0], color=st["color"], marker=st["marker"], lw=2, markersize=7,
                  markeredgecolor="white", label=label)


def plot_best_so_far(searches: list[SearchResults], metric: str, out_path: str | Path, mode: str = "max") -> Path:
    """Best score found so far vs hours spent, one line per method; dots = each trial's own score."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    handles = []
    for s in searches:
        st, curve = METHOD_STYLE[s.method], best_so_far(s, metric, mode)
        ax.step(curve["hours"], curve["best"] * 100, where="post", color=st["color"], lw=2)
        ax.scatter(curve["hours"], curve["value"] * 100, color=st["color"], marker=st["marker"], **DOT)
        handles.append(_legend_handle(
            s.method, f"{st['label']}: best {curve['best'].iloc[-1]:.1%} ({len(curve)} trials)"))

    ax.set_xlabel("Hours spent on successful trials")
    ax.set_ylabel("Val accuracy (%)")
    ax.set_title("Best val accuracy found so far: grid vs random search", loc="left", pad=30)
    ax.grid(axis="y")
    ax.set_xlim(left=0)
    # Legend above the plot, so it never hides a trial
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(handles), borderaxespad=0.3)
    fig.text(0.0, -0.03, "Line = best trial so far. Dots = each trial's own best val accuracy. "
             "Failed attempts are not counted.", fontsize=8.5, color=INK_SECONDARY)
    fig.tight_layout()
    return _save(fig, out_path)


def _log_scale(key: str, searches: list[SearchResults]) -> bool:
    """Use a log axis for a setting that random search samples log-uniformly (learning rates, weight decay)."""
    return any(isinstance(s.space.get(key), dict) and s.space[key].get("dist") == "log_uniform" for s in searches)


def plot_settings_effect(searches: list[SearchResults], metric: str, out_path: str | Path) -> Path:
    """One panel per searched setting: each successful trial's score vs that setting's value.

    A setting that matters shows a trend; one that does not shows a flat cloud.
    """
    keys = list(dict.fromkeys(k for s in searches for k in s.space))  # every searched setting, in config order
    fig, axes = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 4), sharey=True, squeeze=False)
    for ax, key in zip(axes[0], keys):
        for s in searches:
            if key in s.space:
                st = METHOD_STYLE[s.method]
                ax.scatter(s.ok[key], s.ok[metric] * 100, color=st["color"], marker=st["marker"], **DOT)
        not_searched = [METHOD_STYLE[s.method]["label"] for s in searches if key not in s.space]
        if not_searched:
            ax.text(0.03, 0.03, f"{', '.join(not_searched)}: kept fixed", transform=ax.transAxes,
                    fontsize=8, color=INK_SECONDARY)
        if _log_scale(key, searches):
            ax.set_xscale("log")
            # Ticks at 1 and 3 of every power of ten, written like the configs (3e-5, 1e-4, ...)
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 3.0)))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0e}".replace("e-0", "e-")))
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(key + (" (log scale)" if _log_scale(key, searches) else ""))
        ax.grid(axis="y")
    axes[0][0].set_ylabel("Trial's best val accuracy (%)")

    fig.legend(handles=[_legend_handle(s.method, METHOD_STYLE[s.method]["label"]) for s in searches],
               loc="upper right", ncol=len(searches), bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Which setting matters? Each trial's val accuracy vs each searched setting",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    return _save(fig, out_path)
