"""Grid search and random search over training settings (Task 1.2 hyperparameter tuning).

A search = a base training config (e.g. configs/cls_transfer_stage2.yaml) + a search space
(configs/tuning/*.yaml). One trial is a dict {override key: value}, for example
    {"optim.backbone_lr": 1e-4, "optim.weight_decay": 5e-4, "model.dropout": 0.3}
applied with the same `key=value` overrides as `cli train --override`. So every trial is an
ordinary training run, and any trial can be re-run by hand from its overrides.

Grid search
    Every combination of a few listed values (3 x 2 x 2 = 12 trials). Thorough on those values,
    but each setting only ever takes the 3 (or 2) values we listed.
Random search
    n_trials settings drawn at random from ranges. With the same budget, every trial tries a new
    value of every setting, so each range is covered more finely than by the grid.
    Learning rates and weight decay are drawn log-uniformly: 1e-5..1e-4 gets as many trials as
    1e-4..1e-3, because these settings matter by orders of magnitude, not by +1e-4 steps.
"""

import itertools
import math

import numpy as np

from src.utils.config import TuneConfig

DISTRIBUTIONS = ("uniform", "log_uniform")
ABBREVIATIONS = {"backbone_lr": "blr", "weight_decay": "wd", "dropout": "do"}  # run names only


def grid_trials(space: dict[str, list]) -> list[dict[str, object]]:
    """Every combination of the listed values, in a fixed order (the last key changes fastest)."""
    for key, values in space.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Grid search: {key} needs a non-empty list of values, got {values!r}")
    keys = list(space)
    return [dict(zip(keys, combo)) for combo in itertools.product(*space.values())]


def _check_range(key: str, spec: object) -> tuple[str, float, float]:
    """Validate one random-search range {dist, low, high}; returns (dist, low, high)."""
    if not isinstance(spec, dict) or set(spec) != {"dist", "low", "high"}:
        raise ValueError(f"Random search: {key} needs {{dist, low, high}}, got {spec!r}")
    dist, low, high = spec["dist"], float(spec["low"]), float(spec["high"])
    if dist not in DISTRIBUTIONS:
        raise ValueError(f"Random search: {key} dist must be one of {DISTRIBUTIONS}, got {dist!r}")
    if not low < high:
        raise ValueError(f"Random search: {key} needs low < high, got {low} and {high}")
    if dist == "log_uniform" and low <= 0:
        raise ValueError(f"Random search: {key} log_uniform needs low > 0, got {low}")
    return dist, low, high


def random_trials(space: dict[str, dict], n: int, seed: int, digits: int = 3) -> list[dict[str, float]]:
    """n settings, every value drawn independently from its range by a seeded generator.

    log_uniform draws the exponent uniformly: exp(uniform(log low, log high)).
    Values are rounded to `digits` significant digits (2.37e-05, not 2.3698412e-05), so the
    settings in run names, results.csv and the report are exactly the ones that were trained.
    """
    ranges = {key: _check_range(key, spec) for key, spec in space.items()}
    rng = np.random.default_rng(seed)
    trials = []
    for _ in range(n):
        trial = {}
        for key, (dist, low, high) in ranges.items():
            if dist == "log_uniform":
                value = math.exp(rng.uniform(math.log(low), math.log(high)))
            else:
                value = rng.uniform(low, high)
            trial[key] = min(max(float(f"{value:.{digits}g}"), low), high)  # rounding stays in range
        trials.append(trial)
    return trials


def make_trials(cfg: TuneConfig) -> list[dict[str, object]]:
    """All trials of a search, in the order they will run."""
    if cfg.method == "grid":
        return grid_trials(cfg.space)
    if cfg.method == "random":
        if not cfg.n_trials or cfg.n_trials < 1:
            raise ValueError("Random search needs n_trials >= 1 in the tuning config")
        return random_trials(cfg.space, cfg.n_trials, cfg.sampler_seed)
    raise ValueError(f"method must be 'grid' or 'random', got {cfg.method!r}")


def trial_overrides(params: dict[str, object]) -> list[str]:
    """{"optim.lr": 0.001} -> ["optim.lr=0.001"]: the same format as `cli train --override`."""
    return [f"{key}={value}" for key, value in params.items()]


def _short(value: object) -> str:
    """Compact number for run names: 0.0001 -> 1e-4, 2.37e-05 -> 2.37e-5, 0.3 -> 0.3."""
    if not isinstance(value, float):
        return str(value)
    if value == 0 or 0.01 <= abs(value) < 1000:
        return f"{value:.3g}"
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"


def trial_name(method: str, number: int, params: dict[str, object]) -> str:
    """W&B run name, e.g. grid-03-blr1e-4-wd5e-4-do0.3: search, trial number, then the tried values."""
    parts = [f"{ABBREVIATIONS.get(key.split('.')[-1], key.split('.')[-1])}{_short(v)}" for key, v in params.items()]
    return "-".join([method, f"{number:02d}", *parts])
