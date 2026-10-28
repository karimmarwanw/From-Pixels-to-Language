"""Task model for classification = any Encoder + ClassificationHead."""

from torch import Tensor, nn

from src.models.encoders.base import Encoder
from src.models.heads.classification import ClassificationHead


class Classifier(nn.Module):
    """(B, 3, H, W) images -> (B, num_classes) logits. The head only sees encoder.pooled."""

    def __init__(self, encoder: Encoder, head: ClassificationHead) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.encoder(x).pooled)
