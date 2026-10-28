"""From-scratch CNN encoder: the Task 1.2 baseline (random init, no pretrained weights).

VGG-style stages: each stage is `convs_per_stage` x (Conv3x3 -> BatchNorm -> ReLU),
then MaxPool 2x2. With the default 5 stages and 224x224 input:

    stage    channels   output size
      0          32       112 x 112
      1          64        56 x 56
      2         128        28 x 28
      3         256        14 x 14
      4         512         7 x 7   -> global average pool -> (B, 512)

Design choices (explain these in the discussion):
- 3x3 convs stacked: two 3x3 convs see a 5x5 region with fewer parameters than one 5x5 conv.
- BatchNorm after each conv: stabilizes training from scratch and allows a larger learning rate.
  The conv has no bias because BatchNorm's shift (beta) makes it redundant.
- Channels double when resolution halves: keeps the compute per stage roughly constant.
- Global average pooling instead of flatten + big FC: 512 numbers instead of 512*7*7, so far
  fewer parameters to overfit with ~25 images per class. It also keeps spatial maps for Grad-CAM.
- Kaiming (He) init: the standard init for ReLU networks, keeps activations from shrinking/exploding.
- Features at 5 scales (strides 2..32) are what the M2 U-Net needs as skip connections.
"""

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from src.models.encoders.base import Encoder, EncoderOutput


def conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    """(B, in_ch, H, W) -> (B, out_ch, H, W)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ScratchCNN(Encoder):
    def __init__(
        self, channels: Sequence[int] = (32, 64, 128, 256, 512), convs_per_stage: int = 2, in_channels: int = 3
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.stage_channels = list(channels)
        self.out_dim = self.stage_channels[-1]

        stages, prev = [], in_channels
        for ch in self.stage_channels:
            blocks = [conv_bn_relu(prev if j == 0 else ch, ch) for j in range(convs_per_stage)]
            stages.append(nn.Sequential(*blocks, nn.MaxPool2d(2)))  # halves H and W
            prev = ch
        self.stages = nn.ModuleList(stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> EncoderOutput:
        """x: (B, in_channels, H, W) -> features [(B, C_i, H/2^(i+1), W/2^(i+1))], pooled (B, out_dim)."""
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        pooled = torch.flatten(self.pool(x), 1)
        return EncoderOutput(features=features, pooled=pooled)
