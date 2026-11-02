"""Tests for the training-curve figures (src/visualization/curves.py) on tiny made-up histories."""

import pandas as pd
import pytest

from src.visualization.curves import (
    _spread, best_epoch, comparison_table, load_history, plot_comparison, plot_run_curves, run_label)


def _history(val: list[float], train: list[float]) -> pd.DataFrame:
    n = len(val)
    return pd.DataFrame({"epoch": range(1, n + 1), "train_loss": [2.0 - 0.1 * i for i in range(n)],
                         "train_acc": train, "val_loss": [2.2 - 0.05 * i for i in range(n)],
                         "val_acc": val, "val_top5": [min(1.0, v + 0.2) for v in val]})


def test_run_label_and_load_history(tmp_path):
    d = tmp_path / "resnet50-stage2-final_20261029-192701"
    d.mkdir()
    _history([0.5, 0.6], [0.6, 0.8]).to_csv(d / "history.csv", index=False)
    assert run_label(d / "history.csv") == "resnet50-stage2-final"
    assert list(load_history(d / "history.csv")["val_acc"]) == [0.5, 0.6]


def test_best_epoch_takes_the_first_of_equal_scores():
    assert best_epoch(_history([0.5, 0.7, 0.7, 0.6], [0.6, 0.8, 0.9, 0.95])) == (2, 0.7)


def test_comparison_table_reports_the_best_epoch_and_gap():
    t = comparison_table({"a": _history([0.5, 0.7, 0.6], [0.6, 0.9, 0.95])}).iloc[0]
    assert (t["epochs"], t["best_epoch"], t["best_val_acc"]) == (3, 2, 0.7)
    assert t["gap"] == pytest.approx(0.2)  # train 0.9 - val 0.7 at the best epoch


def test_labels_are_spread_apart_in_order():
    assert _spread([10.0, 10.5, 30.0], min_gap=2.0) == [10.0, 12.0, 30.0]


def test_curve_figures_are_written(tmp_path):
    hs = {"scratch": _history([0.1, 0.3, 0.4], [0.1, 0.4, 0.6]), "transfer": _history([0.6, 0.8], [0.7, 0.99])}
    for path in (plot_run_curves(hs["scratch"], "scratch", tmp_path / "a.png"),
                 plot_comparison(hs, tmp_path / "b.png")):
        assert path.exists() and path.stat().st_size > 0
