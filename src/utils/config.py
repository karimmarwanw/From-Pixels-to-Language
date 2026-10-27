"""Load YAML config files into typed dataclasses."""

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class EDAConfig:
    train_name: str
    cache_dir: str
    figures_dir: str
    stats_dir: str
    seed: int = 42
    n_inspect: int = 3
    rebuild_stats: bool = False
    small_side_px: int = 224
    gray_tolerance: int = 3
    n_unusual: int = 5
    wide_aspect: float = 2.0
    tall_aspect: float = 0.5
    dark_brightness: float = 35
    bright_brightness: float = 220
    dup_max_distance: int = 4
    n_lookalikes: int = 4
    n_samples: int = 16
    sample_cols: int = 4
    similar_groups: list[str] = field(default_factory=lambda: ["Gull", "Tern"])
    caption_wrap: int = 40


def load_eda_config(path: str | Path) -> EDAConfig:
    """Read a YAML file into an EDAConfig. Unknown keys raise an error (catches typos)."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown = set(raw) - {f.name for f in fields(EDAConfig)}
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}")
    return EDAConfig(**raw)
