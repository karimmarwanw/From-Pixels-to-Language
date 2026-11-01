"""Full classification metrics, computed from saved predictions (the .npz that `evaluate` writes).

What each metric measures (for the report):
    top-1 accuracy   Share of images whose highest-scoring class is the true species.
    top-5 accuracy   Share whose true species is among the 5 highest scores. With 200 look-alike
                     species, a model that mixes up two warblers still ranks the right one high.
    macro precision  Per species: of the images the model CALLED that species, the share that
                     really are it. Then the plain average over the 200 species. A low value for
                     one species means the model over-uses that label.
    macro recall     Per species: of the images that ARE that species, the share it found.
                     Averaged the same way. A low value means the species is often missed.
    macro F1         Per species: the harmonic mean of its precision and recall, then averaged.
                     "Macro" gives every species the same weight, however many images it has.
                     CUB is nearly balanced, so macro recall stays close to accuracy; a big gap
                     between macro F1 and accuracy would mean a few species fail badly.
    confusion matrix cm[i, j] = number of images of true species i predicted as species j.
    confused pairs   The largest off-diagonal entries: "true A predicted as B", by count.
    within-group     Share of all mistakes where the predicted species is in the same group as
    error share      the true one (e.g. a warbler called another warbler). The group is the last
                     word of the species name, a simple heuristic that matches CUB's naming.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.training.classification import topk_accuracy


def load_predictions(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Logits (N, C), labels (N,) and the C class names from an `evaluate` .npz file."""
    data = np.load(path)
    return data["logits"], data["labels"], [str(c) for c in data["class_names"]]


def species_group(name: str) -> str:
    """'Black throated Blue Warbler' -> 'Warbler'."""
    return name.split()[-1]


def summary_metrics(logits: np.ndarray, labels: np.ndarray, class_names: list[str]) -> dict[str, float]:
    """top1, top5, macro_precision, macro_recall, macro_f1, within_group_error_share, n."""
    preds = logits.argmax(axis=1)  # (N,)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=np.arange(len(class_names)), average="macro", zero_division=0)
    wrong = preds != labels
    groups = np.array([species_group(n) for n in class_names])
    within = (groups[preds[wrong]] == groups[labels[wrong]]).mean() if wrong.any() else 0.0
    t_logits, t_labels = torch.from_numpy(logits), torch.from_numpy(labels)
    return {
        "top1": topk_accuracy(1)(t_logits, t_labels),
        "top5": topk_accuracy(5)(t_logits, t_labels),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
        "within_group_error_share": float(within),
        "n": int(len(labels)),
    }


def per_class_metrics(logits: np.ndarray, labels: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    """One row per species: class, group, n (true images), precision, recall, f1."""
    p, r, f1, support = precision_recall_fscore_support(
        labels, logits.argmax(axis=1), labels=np.arange(len(class_names)), average=None, zero_division=0)
    return pd.DataFrame({"class": class_names, "group": [species_group(n) for n in class_names],
                         "n": support, "precision": p, "recall": r, "f1": f1})


def confusion(logits: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    """(C, C) counts: rows = true species, columns = predicted species."""
    return confusion_matrix(labels, logits.argmax(axis=1), labels=np.arange(n_classes))


def most_confused_pairs(cm: np.ndarray, class_names: list[str], n: int = 15) -> pd.DataFrame:
    """The n largest off-diagonal counts, largest first (ties: lower true index first).

    Columns: true, predicted, count, true_total (images of the true species), share (count / true_total).
    """
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(-off, axis=None, kind="stable")[:n]  # stable sort: ties keep row-major order
    rows, cols = np.unravel_index(flat, off.shape)
    keep = off[rows, cols] > 0
    rows, cols = rows[keep], cols[keep]
    totals = cm.sum(axis=1)
    return pd.DataFrame({
        "true": [class_names[i] for i in rows],
        "predicted": [class_names[j] for j in cols],
        "count": off[rows, cols],
        "true_total": totals[rows],
        "share": off[rows, cols] / np.maximum(totals[rows], 1),
    })
