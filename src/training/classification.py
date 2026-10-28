"""Classification pieces for the generic Trainer: the loss and the metrics tracked every epoch.

These are running training/validation metrics only. The full evaluation (macro precision,
recall, F1, confusion matrix, most-confused pairs) lives in src/evaluation/.
"""

from collections.abc import Callable

import torch
from torch import Tensor, nn


def topk_accuracy(k: int) -> Callable[[Tensor, Tensor], float]:
    """Metric: share of images whose true class is among the k highest logits.

    top-1 = ordinary accuracy. top-5 is informative for 200 fine-grained classes: a model
    that confuses two similar warblers still ranks the right one near the top.
    """

    def metric(logits: Tensor, targets: Tensor) -> float:  # logits (B, C), targets (B,)
        topk = logits.topk(min(k, logits.shape[1]), dim=1).indices  # (B, k)
        return (topk == targets.unsqueeze(1)).any(dim=1).float().mean().item()

    return metric


CLASSIFICATION_METRICS = {"acc": topk_accuracy(1), "top5": topk_accuracy(5)}


def classification_loss(label_smoothing: float = 0.0) -> nn.Module:
    """Cross-entropy on raw logits (softmax is inside the loss).

    label_smoothing=e trains toward (1 - e) on the true class and e spread over all classes,
    so the model is less over-confident; useful with ~25 training images per class.
    """
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


@torch.no_grad()
def predict_logits(model: nn.Module, loader, device: torch.device) -> tuple[Tensor, Tensor]:
    """Run a model over a loader. Returns (logits (N, C), labels (N,)) on the CPU."""
    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        all_logits.append(model(x.to(device, non_blocking=True)).float().cpu())
        all_labels.append(y)
    return torch.cat(all_logits), torch.cat(all_labels)
