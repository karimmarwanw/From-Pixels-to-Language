"""Tests for our Grad-CAM (src/visualization/gradcam.py) on a tiny scratch CNN."""

import torch
import torch.nn.functional as F

from src.models.registry import build_classifier
from src.utils.config import ModelConfig
from src.visualization.gradcam import GradCAM


def _model() -> torch.nn.Module:
    torch.manual_seed(0)
    return build_classifier(ModelConfig(encoder={"channels": [8, 16, 32]}), num_classes=5).eval()


def _expected(model, x, relative: bool) -> torch.Tensor:
    """Grad-CAM by hand for encoder -> global average pool -> linear head (dropout is off in eval).

    There, d score / d A_k(i, j) = w_k / (h * w) at every position, so the channel weights are just the
    head weights of the explained class (minus the average class weights when relative).
    """
    with torch.no_grad():
        A = model.encoder(x).features[-1]  # (B, K, h, w)
        pred = model(x).argmax(dim=1)
        W = model.head.fc.weight  # (C, K)
        w = W[pred] - (W.mean(dim=0) if relative else 0)  # (B, K)
        cam = torch.relu(torch.einsum("bk,bkhw->bhw", w, A))[:, None]
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        return cam / cam.flatten(1).amax(dim=1).clamp_min(1e-8)[:, None, None]


def test_maps_have_input_size_and_range_and_hooks_are_removed():
    model, x = _model(), torch.randn(3, 3, 64, 64)
    layer = model.encoder.gradcam_layer
    with GradCAM(model) as cam:
        maps, logits = cam(x)
    assert maps.shape == (3, 64, 64) and logits.shape == (3, 5)
    assert maps.min() >= 0 and maps.max() <= 1
    assert len(layer._forward_hooks) == 0  # nothing left attached to the model


def test_maps_match_the_hand_computed_formula():
    model, x = _model(), torch.randn(2, 3, 64, 64)
    for relative in (True, False):
        with GradCAM(model, relative=relative) as cam:
            maps, _ = cam(x)
        assert torch.allclose(maps, _expected(model, x, relative), atol=1e-5)


def test_works_on_a_frozen_encoder():
    model = _model()
    model.encoder.freeze()  # e.g. the stage 1 model: no weight needs a gradient
    with GradCAM(model) as cam:
        maps, _ = cam(torch.randn(1, 3, 64, 64))
    assert maps.max() > 0
