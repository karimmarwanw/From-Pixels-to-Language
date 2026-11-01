"""Tests for the full classification metrics (src/evaluation/metrics.py) on a tiny hand-made example."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    confusion, load_predictions, most_confused_pairs, per_class_metrics, species_group, summary_metrics)

NAMES = ["Cape May Warbler", "Yellow Warbler", "Song Sparrow"]
LABELS = np.array([0, 0, 1, 1, 2, 2])
PREDS = np.array([0, 1, 1, 1, 2, 0])  # 2 mistakes: warbler 0 -> warbler 1, sparrow 2 -> warbler 0


def _logits(preds: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """Logits whose argmax is `preds` (the true class is never in the top 1 of a wrong image)."""
    logits = np.zeros((len(preds), n_classes), dtype=np.float32)
    logits[np.arange(len(preds)), preds] = 5.0
    return logits


def test_summary_metrics_match_hand_computed_values():
    s = summary_metrics(_logits(PREDS), LABELS, NAMES)
    # per-class precision: 1/2, 2/3, 1/1; recall: 1/2, 2/2, 1/2; F1: 0.5, 0.8, 2/3
    assert s["top1"] == pytest.approx(4 / 6)
    assert s["macro_precision"] == pytest.approx((0.5 + 2 / 3 + 1.0) / 3)
    assert s["macro_recall"] == pytest.approx((0.5 + 1.0 + 0.5) / 3)
    assert s["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert s["within_group_error_share"] == pytest.approx(0.5)  # 1 of the 2 mistakes is warbler -> warbler
    assert s["top5"] == 1.0 and s["n"] == 6  # with only 3 classes, every true class is in the top 5


def test_confusion_and_most_confused_pairs():
    cm = confusion(_logits(PREDS), LABELS, 3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]  # rows = true, columns = predicted
    pairs = most_confused_pairs(cm, NAMES, n=5)
    assert len(pairs) == 2  # only real mistakes, no zero-count pairs
    assert pairs.iloc[0][["true", "predicted", "count", "true_total"]].tolist() == \
        ["Cape May Warbler", "Yellow Warbler", 1, 2]  # tie on count: lower true index first
    assert pairs.iloc[1]["true"] == "Song Sparrow" and pairs.iloc[1]["share"] == 0.5


def test_per_class_metrics_and_groups():
    t = per_class_metrics(_logits(PREDS), LABELS, NAMES)
    assert t["n"].tolist() == [2, 2, 2] and t["recall"].tolist() == [0.5, 1.0, 0.5]
    assert t["group"].tolist() == ["Warbler", "Warbler", "Sparrow"]
    assert species_group("Black throated Blue Warbler") == "Warbler"


def test_load_predictions_round_trip(tmp_path):
    p = tmp_path / "run_test.npz"
    np.savez_compressed(p, logits=_logits(PREDS), labels=LABELS, class_names=np.array(NAMES))
    logits, labels, names = load_predictions(p)
    assert logits.shape == (6, 3) and labels.tolist() == LABELS.tolist() and names == NAMES


def test_evaluation_figures_are_written(tmp_path):
    from src.visualization.evaluation import plot_confused_pairs, plot_confusion_matrix

    cm = confusion(_logits(PREDS), LABELS, 3)
    for path in (plot_confused_pairs(most_confused_pairs(cm, NAMES), "t", tmp_path / "a.png"),
                 plot_confusion_matrix(cm, NAMES, "t", tmp_path / "b.png", min_group=2)):
        assert path.exists() and path.stat().st_size > 0
