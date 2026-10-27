"""Load YAML config files into typed dataclasses."""

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class EDAConfig:
    train_name: str
    cache_dir: str
    figures_dir: str
    seed: int = 42
    n_inspect: int = 3


def load_eda_config(path: str | Path) -> EDAConfig:
    """Read a YAML file into an EDAConfig. Unknown keys raise an error (catches typos)."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown = set(raw) - {f.name for f in fields(EDAConfig)}
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}")
    return EDAConfig(**raw)
