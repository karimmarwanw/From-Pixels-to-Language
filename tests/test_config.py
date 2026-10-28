"""Quick tests for the training config loader (src/utils/config.py)."""

import pytest

from src.utils.config import flatten, load_train_config


def test_scratch_config_loads_with_types():
    cfg = load_train_config("configs/cls_scratch.yaml")
    assert cfg.experiment == "cls-scratch"
    assert isinstance(cfg.optim.lr, float) and cfg.optim.lr == pytest.approx(1e-3)
    assert cfg.model.encoder["channels"][-1] == 512


def test_overrides_and_string_floats(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("experiment: e\nrun_name: r\noptim:\n  lr: 1e-3\n")  # PyYAML reads 1e-3 as a string
    cfg = load_train_config(p, ["optim.lr=3e-4", "train.epochs=2", "data.limit=null"])
    assert cfg.optim.lr == pytest.approx(3e-4) and isinstance(cfg.optim.lr, float)
    assert cfg.train.epochs == 2 and cfg.data.limit is None


def test_unknown_key_is_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("experiment: e\nrun_name: r\noptim:\n  learning_rate: 0.1\n")
    with pytest.raises(ValueError, match="learning_rate"):
        load_train_config(p)


def test_flatten():
    assert flatten({"a": {"b": 1, "c": {"d": 2}}, "e": 3}) == {"a.b": 1, "a.c.d": 2, "e": 3}
