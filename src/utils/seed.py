"""Reproducibility: one seed controls python, numpy and torch (CPU + CUDA), and DataLoader workers.

Bit-exact GPU runs also need deterministic cuDNN kernels, which are slower. We default to
deterministic=False (cudnn.benchmark on, faster on the T4) and say so in the report: two runs
with the same seed match in data order and initialization, but may differ in the last digits.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed every random number generator we use."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic  # benchmark picks the fastest kernel (non-deterministic)


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: give each worker its own numpy/random seed derived from torch's.

    Without this, every worker process starts with the same numpy state, so random
    augmentations that use numpy would repeat across workers.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Generator for DataLoader shuffling, so the batch order is the same on every machine."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
