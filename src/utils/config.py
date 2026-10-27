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
    n_top_words: int = 30
    min_freq_options: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])
    stop_words: list[str] = field(default_factory=list)
    color_words: list[str] = field(default_factory=list)
    body_part_words: list[str] = field(default_factory=list)
    typo_examples: list[str] = field(default_factory=list)
    n_template_examples: int = 5
    split_path: str = "data/splits/split.json"
    val_size: float = 0.15
    image_size: int = 224
    stats_path: str = "data/processed/stats.json"
    imagenet_mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    imagenet_std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


def load_eda_config(path: str | Path) -> EDAConfig:
    """Read a YAML file into an EDAConfig. Unknown keys raise an error (catches typos)."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown = set(raw) - {f.name for f in fields(EDAConfig)}
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}")
    return EDAConfig(**raw)
