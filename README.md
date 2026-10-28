# From-Pixels-to-Language

CSE485/CSE488 Deep Learning project on CUB-200-2011: one CNN encoder for bird classification
(Milestone 1), then colorization and captioning (Milestone 2).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                                           # quick checks, no download needed
```

On Kaggle use `notebooks/kaggle_launcher.ipynb` (it installs only what Kaggle is missing).

## Classification (Task 1.2)

```bash
# Smoke test: 1 epoch on 256 images
python -m src.cli train --task classification --config configs/cls_scratch.yaml \
    --override data.limit=256 train.epochs=1 run_name=smoke

# From-scratch CNN baseline
python -m src.cli train --task classification --config configs/cls_scratch.yaml

# Resume an interrupted run (same W&B run, continues after the last finished epoch)
python -m src.cli train --task classification --config configs/cls_scratch.yaml \
    --resume outputs/checkpoints/<run_dir>/last.pt

# Evaluate a checkpoint (val by default; --split test only for the final models)
python -m src.cli evaluate --task classification --checkpoint outputs/checkpoints/<run_dir>/best.pt --split val

# Predict one image
python -m src.cli predict --task classification --checkpoint outputs/checkpoints/<run_dir>/best.pt --image bird.jpg
```

Any config value can be changed from the command line with `--override key=value`,
e.g. `--override optim.lr=3e-4 data.batch_size=32`.

Each run writes `outputs/checkpoints/<run_name>_<timestamp>/` with `best.pt` (best val accuracy),
`last.pt` (for resuming) and `history.csv` (one row per epoch).

## Hyperparameter tuning (Task 1.2)

Grid search and random search over transfer learning stage 2 (12 trials each, 8 epochs per trial).
Every trial starts from the stage 1 `best.pt` and is an ordinary training run, tracked in W&B.

```bash
# Check the setup first: 1 trial, 64 images, 1 epoch, no W&B, results in a separate folder
python -m src.cli tune --task classification --method grid --config configs/tuning/cls_grid.yaml \
    --max-trials 1 --results-dir outputs/tuning-smoke \
    --override data.limit=64 train.epochs=1 tracking.wandb_mode=disabled

# Grid search: every combination in configs/tuning/cls_grid.yaml
python -m src.cli tune --task classification --method grid --config configs/tuning/cls_grid.yaml \
    --override model.init_from=outputs/checkpoints/<stage1_run_dir>/best.pt

# Random search: 12 settings drawn from the ranges in configs/tuning/cls_random.yaml
python -m src.cli tune --task classification --method random --config configs/tuning/cls_random.yaml \
    --override model.init_from=outputs/checkpoints/<stage1_run_dir>/best.pt
```

- `outputs/tuning/<experiment>/results.csv` gets one row per trial: settings, best val accuracy,
  time, W&B run id, and `ok` or `failed`. W&B groups: `cls-tuning-grid`, `cls-tuning-random`.
- Interrupted? Run the same command again: trials already in `results.csv` are skipped.
- Trial checkpoints are deleted after each trial (`keep_checkpoints: false`); `history.csv` stays.
- On a Mac, start long searches with `caffeinate -i python -m src.cli tune ...` so the laptop does not sleep.

## Experiment tracking (Weights & Biases / CoreWeave Forge)

Every run is logged to the team's W&B project, so all members' runs share one dashboard.

```bash
wandb login                      # once per machine; paste your API key from your W&B settings
python -m src.cli train --task classification --config configs/cls_scratch.yaml \
    --override tracking.wandb_entity=<team-name>
```

- Set `tracking.wandb_entity` in the config to the team name (or pass it with `--override`).
- On Kaggle, add your key as a secret named `WANDB_API_KEY` (*Add-ons → Secrets*); the launcher notebook reads it.
- No internet or no account: `--override tracking.wandb_mode=offline`, then `wandb sync wandb/offline-run-*` later.
- Never commit an API key.

## EDA (Task 1.1)

```bash
python -m src.cli eda --config configs/eda.yaml --step N   # N = 1..10, see docs/eda_findings.md
```
