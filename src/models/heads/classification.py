"""Classification head: pooled encoder vector -> class logits."""

from torch import Tensor, nn


class ClassificationHead(nn.Module):
    """(B, in_dim) -> Dropout -> Linear -> (B, num_classes) logits (no softmax; the loss applies it)."""

    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)  # randomly zeroes features in training: less co-adaptation
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, pooled: Tensor) -> Tensor:
        return self.fc(self.dropout(pooled))
