"""Image transforms: random augmentation for training, a fixed pipeline for val/test.

Every image (any size, any mode) leaves as a float tensor (3, image_size, image_size),
normalized per channel with (x - mean) / std. Each augmentation choice comes from the EDA:
    - RandomResizedCrop: images are ~500x375 and the bird's size/position varies (steps 3, 5);
      scale is limited (crop_scale) so the bird is rarely cropped out.
    - RandomHorizontalFlip: birds face left or right equally; a mirrored bird is the same species.
    - small RandomRotation: camera tilt / perched at an angle. NO vertical flip: birds are never upside down.
    - mild brightness/contrast ColorJitter: lighting varies a lot (dark flash shots, white skies; step 3).
      NO hue / saturation change: color is a key cue between look-alike species (steps 5, 6).
Val/test: Resize(short side) + CenterCrop only, so evaluation is deterministic.
"""

from collections.abc import Sequence

from PIL import Image
from torchvision import transforms as T


class ToRGB:
    """Convert any PIL image (L, RGBA, CMYK, ...) to 3-channel RGB. 4 CUB train images are grayscale."""

    def __call__(self, img: Image.Image) -> Image.Image:
        return img.convert("RGB")

    def __repr__(self) -> str:
        return "ToRGB()"


def _to_tensor_and_normalize(mean: Sequence[float], std: Sequence[float]) -> list:
    return [T.ToTensor(), T.Normalize(mean=list(mean), std=list(std))]  # (3, H, W) floats, centred ~0


def build_train_transform(
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    crop_scale: Sequence[float],
    hflip_p: float,
    rotation_deg: float,
    brightness: float,
    contrast: float,
    color_jitter: bool = True,
    to_tensor: bool = True,
) -> T.Compose:
    """Random training augmentation. Output: tensor (3, image_size, image_size).

    Args:
        crop_scale: (min, max) share of the image area kept by RandomResizedCrop.
        hflip_p: probability of a horizontal flip.
        rotation_deg: rotation is drawn from [-rotation_deg, +rotation_deg]. Corners exposed by the
            rotation are filled with the mean color, which becomes 0 after normalization.
        brightness, contrast: ColorJitter strengths (0 = off). Hue and saturation are never changed.
        color_jitter: False gives a geometric-only variant (e.g. for colorization targets in M2).
        to_tensor: False stops before ToTensor/Normalize and returns a PIL image (for plotting).
    """
    fill = tuple(round(m * 255) for m in mean)
    steps = [
        ToRGB(),
        T.RandomResizedCrop(image_size, scale=tuple(crop_scale)),
        T.RandomHorizontalFlip(p=hflip_p),
        T.RandomRotation(degrees=rotation_deg, fill=fill),
    ]
    if color_jitter:
        steps.append(T.ColorJitter(brightness=brightness, contrast=contrast, saturation=0, hue=0))
    if to_tensor:
        steps += _to_tensor_and_normalize(mean, std)
    return T.Compose(steps)


def build_eval_transform(
    image_size: int, resize_size: int, mean: Sequence[float], std: Sequence[float], to_tensor: bool = True
) -> T.Compose:
    """Deterministic val/test transform: RGB -> resize short side to resize_size -> center crop image_size.

    Output: tensor (3, image_size, image_size), or a PIL image if to_tensor is False.
    """
    steps = [ToRGB(), T.Resize(resize_size), T.CenterCrop(image_size)]
    if to_tensor:
        steps += _to_tensor_and_normalize(mean, std)
    return T.Compose(steps)
