"""Quick tests for the image transforms (src/data/transforms.py)."""

import torch
from PIL import Image

from src.data.transforms import build_eval_transform, build_train_transform

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
AUG = dict(crop_scale=(0.5, 1.0), hflip_p=0.5, rotation_deg=15, brightness=0.2, contrast=0.2)


def _image(mode: str = "RGB", size: tuple[int, int] = (500, 375)) -> Image.Image:
    return Image.new(mode, size, color=128 if mode == "L" else (120, 80, 40))


def test_train_output_shape_for_any_size_and_mode():
    tf = build_train_transform(224, MEAN, STD, **AUG)
    for img in [_image(), _image("L"), _image(size=(500, 139)), _image(size=(121, 170))]:
        x = tf(img)
        assert x.shape == (3, 224, 224) and x.dtype == torch.float32


def test_eval_output_shape_for_any_size_and_mode():
    tf = build_eval_transform(224, 256, MEAN, STD)
    for img in [_image(), _image("L"), _image(size=(500, 139)), _image(size=(121, 170))]:
        assert tf(img).shape == (3, 224, 224)


def test_eval_is_deterministic():
    img = Image.effect_noise((500, 375), 50).convert("RGB")
    tf = build_eval_transform(224, 256, MEAN, STD)
    assert torch.equal(tf(img), tf(img))


def test_train_is_random():
    img = Image.effect_noise((500, 375), 50).convert("RGB")
    tf = build_train_transform(224, MEAN, STD, **AUG)
    torch.manual_seed(0)
    assert not torch.equal(tf(img), tf(img))


def test_normalize_uses_mean_and_std():
    # a uniform image exactly at the mean color must become ~0 in every channel
    img = Image.new("RGB", (300, 300), tuple(round(m * 255) for m in MEAN))
    x = build_eval_transform(224, 256, MEAN, STD)(img)
    assert x.abs().max() < 0.02
