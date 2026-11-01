"""Grad-CAM, written with our own hooks: which part of the image made the model pick its class?

Method (Selvaraju et al., ICCV 2017), for one image and one class c (we explain the PREDICTED class):
    1. Forward pass. A forward hook on encoder.gradcam_layer (the last conv stage) keeps its
       output A: (B, K, h, w). ResNet50: K=2048, scratch CNN: K=512; h = w = 7 for 224x224 input.
    2. Backward pass from the class score s_c. A hook on A keeps the gradient ds_c/dA: (B, K, h, w).
       The paper uses the raw logit, s_c = y_c. We use s_c = y_c - mean_j(y_j) by default
       (relative=True): the predicted class COMPARED WITH the average class. Softmax only uses the
       differences between logits, and our final ResNet50 gives every class a large negative logit
       (the predicted logit is < 0 for 98% of val images). With the raw logit, that shared negative
       part dominates every cell, the ReLU below erases it, and 30% of the maps come out blank.
       Subtracting the mean removes only the part shared by all classes. relative=False = the paper.
    3. Channel weights alpha_k = average of dy_c/dA_k over the h*w positions: (B, K).
       A large alpha_k means "feature map k pushes the score of class c up".
    4. CAM = ReLU(sum_k alpha_k * A_k): (B, h, w). The ReLU keeps only evidence FOR class c.
    5. Upsample bilinearly to the input size and divide by the maximum, so each map is in [0, 1].
The maps are blurry because each of the 7x7 cells covers a 32x32-pixel patch of the input.
"""

import logging
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.visualization.eda import INK, INK_SECONDARY, _save  # same look as the EDA figures

logger = logging.getLogger(__name__)
HEATMAP = "inferno"  # perceptually even "heat" scale: dark = no evidence, bright yellow = strongest


class GradCAM:
    """Context manager: hooks are attached on enter and always removed on exit.

    with GradCAM(model) as cam:
        maps, logits = cam(x)  # x (B, 3, H, W) -> maps (B, H, W) in [0, 1], logits (B, num_classes)
    """

    def __init__(self, model: nn.Module, layer: nn.Module | None = None, relative: bool = True) -> None:
        self.model = model
        self.layer = layer if layer is not None else model.encoder.gradcam_layer
        self.relative = relative  # explain y_c - mean(y) instead of the raw logit y_c (see module docstring)
        self._acts: Tensor | None = None
        self._grads: Tensor | None = None

    def __enter__(self) -> "GradCAM":
        self._handle = self.layer.register_forward_hook(self._keep_activations)
        return self

    def __exit__(self, *exc) -> None:
        self._handle.remove()

    def _keep_activations(self, module: nn.Module, inputs: tuple, output: Tensor) -> None:
        self._acts = output  # (B, K, h, w)
        output.register_hook(self._keep_gradients)  # runs during backward with dy_c/dA

    def _keep_gradients(self, grad: Tensor) -> None:
        self._grads = grad  # (B, K, h, w)

    def __call__(self, x: Tensor, class_idx: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """x (B, 3, H, W) -> (maps (B, H, W) in [0, 1], logits (B, C)). class_idx (B,): None = predicted."""
        self.model.eval()  # BatchNorm/Dropout in inference mode: images do not affect each other
        with torch.enable_grad():
            x = x.detach().clone().requires_grad_(True)  # builds the graph even if weights are frozen
            logits = self.model(x)
            idx = logits.argmax(dim=1) if class_idx is None else class_idx
            score = logits.gather(1, idx[:, None])  # (B, 1)
            if self.relative:
                score = score - logits.mean(dim=1, keepdim=True)
            self.model.zero_grad(set_to_none=True)
            score.sum().backward()  # images are independent in eval mode: one backward serves the batch
        weights = self._grads.mean(dim=(2, 3), keepdim=True)  # (B, K, 1, 1)
        cam = torch.relu((weights * self._acts).sum(dim=1, keepdim=True))  # (B, 1, h, w)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]  # (B, H, W)
        cam = cam / cam.flatten(1).amax(dim=1).clamp_min(1e-8)[:, None, None]
        self.model.zero_grad(set_to_none=True)
        return cam.detach(), logits.detach()


def _label(name: str, width: int = 24) -> str:
    return textwrap.fill(name, width)


def plot_gradcam_grid(images: list[np.ndarray], maps: list[np.ndarray], titles: list[str],
                      title: str, out_path: str | Path) -> Path:
    """Row 1: the image the model saw (224x224 crop). Row 2: the same image with the Grad-CAM heatmap."""
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(2.5 * n + 1.0, 5.8), squeeze=False, layout="constrained")
    for j, (img, cam, t) in enumerate(zip(images, maps, titles)):
        axes[0, j].imshow(img)
        axes[0, j].set_title(t, fontsize=7.5, color=INK, loc="left")
        axes[1, j].imshow(img)
        heat = axes[1, j].imshow(cam, cmap=HEATMAP, alpha=0.5, vmin=0, vmax=1)
        for ax in axes[:, j]:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_ylabel("Image", color=INK_SECONDARY)
    axes[1, 0].set_ylabel("Grad-CAM overlay", color=INK_SECONDARY)
    bar = fig.colorbar(heat, ax=axes.ravel().tolist(), shrink=0.45, pad=0.01)  # both rows keep the same width
    bar.set_label("Evidence for the predicted class", fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    return _save(fig, out_path)


def example_titles(true: list[str] | None, pred: list[str], conf: list[float]) -> list[str]:
    """'True: X / Pred: Y (87%)' per image (no 'True' line when the label is unknown)."""
    lines = []
    for i, (p, c) in enumerate(zip(pred, conf)):
        head = f"True: {_label(true[i])}\n" if true is not None else ""
        lines.append(f"{head}Pred: {_label(p)} ({c:.0%})")
    return lines


def explain_checkpoint(ckpt_path: str | Path, out_dir: str | Path, n: int = 6, seed: int | None = None,
                       image: str | Path | None = None, device_name: str = "auto") -> list[Path]:
    """Grad-CAM figures for a trained classifier.

    image given: one figure for that image. Otherwise: n correct + n wrong VAL images (never test),
    drawn at random with `seed` (default: the run's seed), one figure each. Returns the figure paths.
    """
    from PIL import Image

    from src.data.datasets import CUBClassification, build_classification_loaders, eval_transform, get_mean_std
    from src.data.loading import load_cub_split
    from src.data.split import load_split
    from src.data.transforms import build_eval_transform
    from src.training.classification import predict_logits
    from src.training.run import load_classifier
    from src.utils.device import get_device

    device = get_device(device_name)
    model, cfg, names = load_classifier(ckpt_path, device)
    model_tf = eval_transform(cfg.data)  # what the model sees: resize, center crop, normalize
    shown_tf = build_eval_transform(cfg.data.image_size, cfg.data.resize_size, *get_mean_std(cfg.data),
                                    to_tensor=False)  # the same crop, without normalization, for display
    out_dir = Path(out_dir)

    def explain(pils: list, true: list[str] | None, title: str, path: Path) -> Path:
        x = torch.stack([model_tf(p) for p in pils]).to(device)
        with GradCAM(model) as cam:
            maps, logits = cam(x)
        probs = logits.softmax(dim=1)
        conf, pred = probs.max(dim=1)
        titles = example_titles(true, [names[i] for i in pred.tolist()], conf.tolist())
        return plot_gradcam_grid([np.asarray(shown_tf(p)) for p in pils], list(maps.cpu().numpy()),
                                 titles, title, path)

    if image is not None:
        pil = Image.open(image)
        return [explain([pil], None, f"Grad-CAM: {cfg.run_name}", out_dir / f"{cfg.run_name}_{Path(image).stem}.png")]

    hf = load_cub_split(cfg.data.train_name, "train", cfg.data.cache_dir)
    _, val_idx = load_split(cfg.data.split_path)
    val_ds = CUBClassification(hf, val_idx, model_tf)
    _, loader, _ = build_classification_loaders(cfg.data, cfg.seed, hf_train=hf)
    logits, labels = predict_logits(model, loader, device)  # (N, C), (N,) in val order
    correct = (logits.argmax(dim=1) == labels).numpy()
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    paths = []
    for kind, pool in (("correct", np.flatnonzero(correct)), ("wrong", np.flatnonzero(~correct))):
        pick = np.sort(rng.choice(pool, size=min(n, len(pool)), replace=False))
        pils = [val_ds.ds[int(val_ds.indices[i])]["image"] for i in pick]
        true = [names[int(labels[i])] for i in pick]
        title = f"Grad-CAM, {kind} predictions: {cfg.run_name} (val images, explained class = predicted)"
        paths.append(explain(pils, true, title, out_dir / f"{cfg.run_name}_val_{kind}.png"))
        logger.info("%s: %d of %d val images; picked %s", kind, len(pool), len(correct), pick.tolist())
    return paths
