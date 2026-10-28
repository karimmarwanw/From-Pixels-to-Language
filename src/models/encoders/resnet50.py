"""Pretrained ResNet50 encoder: the Task 1.2 transfer learning model.

torchvision's ResNet50, cut into 5 stages so it fits the same Encoder interface as the scratch CNN.
With a 224x224 input:

    stage    ResNet50 parts               channels   output size
      0      conv1 -> bn1 -> relu             64     112 x 112
      1      maxpool -> layer1               256      56 x 56
      2      layer2                          512      28 x 28
      3      layer3                         1024      14 x 14
      4      layer4                         2048       7 x 7   -> global average pool -> (B, 2048)

Design choices (explain these in the discussion):
- Pretrained on ImageNet (1.2M photos, 1000 classes): edges, textures, feathers and shapes are
  already learned, so ~25 images per bird species are enough to learn the species themselves.
- Same 5 output sizes as the scratch CNN: freeze/unfreeze, Grad-CAM (last stage) and the M2 U-Net
  skip connections work the same for both encoders.
- ResNet's own fc layer (1000 ImageNet classes) is dropped; our ClassificationHead (200 classes)
  sits on `pooled` instead.
- Inputs must use ImageNet normalization (data.normalization: imagenet), the statistics the
  weights were trained with.
"""

import torch
from torch import Tensor, nn
from torchvision.models import ResNet50_Weights, resnet50

from src.models.encoders.base import Encoder, EncoderOutput


class ResNet50Encoder(Encoder):
    def __init__(self, pretrained: bool = True, in_channels: int = 3) -> None:
        super().__init__()
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        if in_channels != 3:  # e.g. 1 for grayscale colorization input in M2
            net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.in_channels = in_channels
        self.stage_channels = [64, 256, 512, 1024, 2048]
        self.out_dim = 2048
        self.stages = nn.ModuleList([
            nn.Sequential(net.conv1, net.bn1, net.relu),
            nn.Sequential(net.maxpool, net.layer1),
            net.layer2,
            net.layer3,
            net.layer4,
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: Tensor) -> EncoderOutput:
        """x: (B, in_channels, H, W) -> features [(B, C_i, H/2^(i+1), W/2^(i+1))], pooled (B, 2048)."""
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        pooled = torch.flatten(self.pool(x), 1)
        return EncoderOutput(features=features, pooled=pooled)
