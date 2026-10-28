"""Tests for the grid and random search trial generators (src/tuning/search.py)."""

import math

import pytest

from src.tuning.search import grid_trials, make_trials, random_trials, trial_name, trial_overrides
from src.utils.config import load_train_config, load_tune_config

GRID = "configs/tuning/cls_grid.yaml"
RANDOM = "configs/tuning/cls_random.yaml"


def test_grid_config_gives_12_different_trials():
    trials = make_trials(load_tune_config(GRID))
    assert len(trials) == 12
    assert len({tuple(sorted(t.items())) for t in trials}) == 12  # no combination twice
    assert {t["optim.backbone_lr"] for t in trials} == {3e-5, 1e-4, 3e-4}


def test_grid_order_last_key_changes_fastest():
    trials = grid_trials({"a": [1, 2], "b": [10, 20]})
    assert trials == [{"a": 1, "b": 10}, {"a": 1, "b": 20}, {"a": 2, "b": 10}, {"a": 2, "b": 20}]


def test_random_config_is_in_range_and_reproducible():
    cfg = load_tune_config(RANDOM)
    trials = make_trials(cfg)
    assert len(trials) == 12 and trials == make_trials(cfg)  # same seed -> same 12 settings
    for t in trials:
        assert set(t) == set(cfg.space)
        for key, spec in cfg.space.items():
            assert float(spec["low"]) <= t[key] <= float(spec["high"])


def test_random_seed_changes_the_settings():
    space = {"x": {"dist": "uniform", "low": 0.0, "high": 1.0}}
    assert random_trials(space, 5, seed=0) != random_trials(space, 5, seed=1)


def test_log_uniform_gives_each_order_of_magnitude_equal_weight():
    space = {"lr": {"dist": "log_uniform", "low": 1e-5, "high": 1e-3}}
    values = [t["lr"] for t in random_trials(space, 2000, seed=0)]
    below = sum(v < 1e-4 for v in values) / len(values)  # 1e-4 = middle on a log scale
    assert 0.45 < below < 0.55  # a plain uniform draw would put only ~9% below 1e-4


def test_random_values_are_rounded_to_3_significant_digits():
    space = {"lr": {"dist": "log_uniform", "low": 1e-5, "high": 1e-3}}
    for t in random_trials(space, 50, seed=0):
        assert t["lr"] == float(f"{t['lr']:.3g}")


@pytest.mark.parametrize("spec", [
    {"dist": "normal", "low": 0.1, "high": 0.5},          # unknown distribution
    {"dist": "uniform", "low": 0.5, "high": 0.1},         # low > high
    {"dist": "log_uniform", "low": 0.0, "high": 1e-3},    # log(0) is undefined
    {"dist": "uniform", "low": 0.1},                      # missing high
])
def test_bad_random_ranges_are_rejected(spec):
    with pytest.raises(ValueError):
        random_trials({"x": spec}, 1, seed=0)


def test_unknown_tune_config_key_is_rejected(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("base_config: b.yaml\nexperiment: e\nmethod: grid\nspace: {}\nn_trial: 3\n")
    with pytest.raises(ValueError, match="n_trial"):
        load_tune_config(p)


def test_trial_overrides_load_into_the_training_config():
    tune = load_tune_config(GRID)
    params = {"optim.backbone_lr": 3e-5, "optim.weight_decay": 5e-3, "model.dropout": 0.5}
    cfg = load_train_config(tune.base_config, trial_overrides(tune.fixed) + trial_overrides(params))
    assert isinstance(cfg.optim.backbone_lr, float) and math.isclose(cfg.optim.backbone_lr, 3e-5)
    assert cfg.optim.weight_decay == pytest.approx(5e-3) and cfg.model.dropout == 0.5
    assert cfg.train.epochs == 8  # from `fixed`
    assert cfg.optim.lr == pytest.approx(1e-3)  # not tuned by the grid: stays as in the base config


def test_trial_name():
    params = {"optim.backbone_lr": 1e-4, "optim.weight_decay": 5e-4, "model.dropout": 0.3}
    assert trial_name("grid", 3, params) == "grid-03-blr1e-4-wd5e-4-do0.3"
    assert trial_name("random", 12, {"optim.lr": 0.00237}) == "random-12-lr2.37e-3"
