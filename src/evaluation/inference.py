"""Run a trained classifier checkpoint: evaluate on val/test, or predict one image.

`evaluate_checkpoint` saves raw predictions (logits + labels) to an .npz file, so the full
metrics (src/evaluation/metrics.py), confusion matrix and Grad-CAM picks of correct/wrong
images can be computed later without re-running the model.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.datasets import build_classification_loaders, build_test_loader, eval_transform
from src.training.classification import CLASSIFICATION_METRICS, predict_logits
from src.training.run import load_classifier
from src.utils.device import get_device

logger = logging.getLogger(__name__)


def evaluate_checkpoint(ckpt_path: str | Path, split: str = "val", out_dir: str | Path = "outputs/predictions",
                        device_name: str = "auto") -> dict:
    """Top-1/top-5 accuracy and cross-entropy on `split` ("val" or "test"); writes <out_dir>/<run>_<split>.npz.

    The test split is for FINAL numbers only: never pick models or settings by test results.
    """
    device = get_device(device_name)
    model, cfg, class_names = load_classifier(ckpt_path, device)
    cfg.data.limit = None  # always evaluate on the full split
    if split == "val":
        _, loader, _ = build_classification_loaders(cfg.data, cfg.seed, pin_memory=device.type == "cuda")
    elif split == "test":
        logger.warning("Evaluating on the TEST set: final evaluation only, not for model selection.")
        loader, test_names = build_test_loader(cfg.data, cfg.seed, pin_memory=device.type == "cuda")
        if test_names != class_names:
            raise ValueError("Train and test label names differ; labels would not line up")
    else:
        raise ValueError(f"split must be 'val' or 'test', got {split!r}")

    logits, labels = predict_logits(model, loader, device)  # (N, C), (N,)
    results = {name: fn(logits, labels) for name, fn in CLASSIFICATION_METRICS.items()}
    results["loss"] = torch.nn.functional.cross_entropy(logits, labels).item()  # plain CE, no smoothing
    results["n"] = len(labels)

    run_dir = Path(ckpt_path).parent.name
    out = Path(out_dir) / f"{run_dir}_{split}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, logits=logits.numpy(), labels=labels.numpy(), class_names=np.array(class_names))
    results["predictions_file"] = str(out)

    _log_to_original_run(ckpt_path, cfg.tracking.tracking_uri, split, results)
    return results


def _log_to_original_run(ckpt_path: str | Path, tracking_uri: str, split: str, results: dict) -> None:
    """Add the evaluation numbers to the MLflow run that trained this checkpoint, if it is on this machine."""
    from src.training.tracking import MLflowTracker

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    try:
        with MLflowTracker(tracking_uri, state["config"]["experiment"], "", run_id=state["mlflow_run_id"]) as t:
            t.log_metrics({f"{split}_eval_{k}": v for k, v in results.items() if isinstance(v, (int, float))})
    except Exception as e:  # e.g. checkpoint trained on Kaggle, evaluated on a laptop with another mlflow.db
        logger.warning("Could not log to the original MLflow run (%s); results are printed only.", e)


@torch.no_grad()
def predict_image(ckpt_path: str | Path, image_path: str | Path, topk: int = 5,
                  device_name: str = "auto") -> list[tuple[str, float]]:
    """Top-k (species, probability) for one image file, using the eval transform (resize + center crop)."""
    device = get_device(device_name)
    model, cfg, class_names = load_classifier(ckpt_path, device)
    x = eval_transform(cfg.data)(Image.open(image_path)).unsqueeze(0).to(device)  # (1, 3, H, W)
    probs = torch.softmax(model(x).float(), dim=1)[0]  # (C,)
    top = probs.topk(min(topk, len(class_names)))
    return [(class_names[i], p) for p, i in zip(top.values.tolist(), top.indices.tolist())]
