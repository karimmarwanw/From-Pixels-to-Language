"""Pick the compute device: CUDA (Kaggle/Colab T4) > Apple MPS (M-series laptops) > CPU."""

import logging

import torch

logger = logging.getLogger(__name__)


def get_device(name: str = "auto") -> torch.device:
    """Return the device named in the config, or the best available one for "auto"."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA: %s", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU (training will be slow)")
    return device
