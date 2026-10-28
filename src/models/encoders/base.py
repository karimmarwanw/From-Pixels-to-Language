"""The shared Encoder interface (CLAUDE.md "Architecture contract").

Every encoder (scratch CNN, ResNet50, ...) returns the same EncoderOutput, so the
classification head now, and the U-Net decoder and caption decoder in Milestone 2,
work with any encoder without changes.

An encoder is a list of `stages`. Each stage lowers the resolution and its output is
one entry of `features` (shallow -> deep). freeze/unfreeze work on whole stages.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from torch import Tensor, nn


@dataclass
class EncoderOutput:
    features: list[Tensor]  # one map per stage, shallow -> deep: (B, stage_channels[i], H_i, W_i)
    pooled: Tensor  # (B, out_dim): global average pool of the deepest map


class Encoder(nn.Module, ABC):
    """Base class. Subclasses set in_channels, stage_channels, out_dim and build self.stages."""

    in_channels: int
    stage_channels: list[int]
    out_dim: int
    stages: nn.ModuleList

    def __init__(self) -> None:
        super().__init__()
        self._frozen: set[int] = set()  # indices of frozen stages

    @property
    def gradcam_layer(self) -> nn.Module:
        """Last stage: deepest feature map, where Grad-CAM hooks attach."""
        return self.stages[-1]

    @abstractmethod
    def forward(self, x: Tensor) -> EncoderOutput:
        """x: (B, in_channels, H, W) -> EncoderOutput."""

    def freeze(self, stages: Sequence[int] | None = None) -> None:
        """Stop training the given stages (all if None): no gradients, BatchNorm stats fixed."""
        for i in range(len(self.stages)) if stages is None else stages:
            for p in self.stages[i].parameters():
                p.requires_grad = False
            self._frozen.add(i)
        self.train(self.training)  # re-apply eval mode to the frozen stages

    def unfreeze(self, stages: Sequence[int] | None = None) -> None:
        """Train the given stages again (all if None)."""
        for i in range(len(self.stages)) if stages is None else stages:
            for p in self.stages[i].parameters():
                p.requires_grad = True
            self._frozen.discard(i)
        self.train(self.training)

    def train(self, mode: bool = True) -> "Encoder":
        """Like nn.Module.train, but frozen stages always stay in eval mode.

        Without this, model.train() would put frozen BatchNorm layers back in training mode
        and they would keep updating their running mean/var, so a "frozen" encoder changes.
        """
        super().train(mode)
        for i in self._frozen:
            self.stages[i].eval()
        return self
