"""Shape and interface tests for the encoder, head and classifier."""

import torch

from src.models.encoders.scratch_cnn import ScratchCNN
from src.models.registry import build_classifier
from src.utils.config import ModelConfig


def test_scratch_encoder_output_shapes():
    enc = ScratchCNN()
    out = enc(torch.randn(2, 3, 224, 224))
    assert [f.shape for f in out.features] == [
        (2, 32, 112, 112), (2, 64, 56, 56), (2, 128, 28, 28), (2, 256, 14, 14), (2, 512, 7, 7)
    ]
    assert out.pooled.shape == (2, enc.out_dim) == (2, 512)
    assert enc.stage_channels == [32, 64, 128, 256, 512]


def test_single_channel_input_for_colorization():
    enc = ScratchCNN(channels=[8, 16], in_channels=1)
    assert enc(torch.randn(1, 1, 64, 64)).pooled.shape == (1, 16)


def test_classifier_from_config():
    model = build_classifier(ModelConfig(encoder={"channels": [8, 16, 32]}), num_classes=200)
    assert model(torch.randn(3, 3, 64, 64)).shape == (3, 200)


def test_freeze_keeps_batchnorm_fixed_and_unfreeze_restores():
    enc = ScratchCNN(channels=[8, 16])
    enc.freeze([0])
    enc.train()
    assert all(not p.requires_grad for p in enc.stages[0].parameters())
    assert all(p.requires_grad for p in enc.stages[1].parameters())
    bn = enc.stages[0][0][1]
    before = bn.running_mean.clone()
    enc(torch.randn(4, 3, 32, 32))
    assert not bn.training and torch.equal(bn.running_mean, before)  # stats did not move
    enc.unfreeze()
    enc.train()
    assert bn.training and all(p.requires_grad for p in enc.parameters())


def test_gradcam_layer_is_last_stage():
    enc = ScratchCNN(channels=[8, 16])
    assert enc.gradcam_layer is enc.stages[-1]
