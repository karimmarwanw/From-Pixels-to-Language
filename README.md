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

# Resume an interrupted run (same MLflow run, continues after the last finished epoch)
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

## Experiment tracking

Runs are logged to MLflow in a local SQLite file (`mlflow.db`, artifacts in `mlruns/`):

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db    # then open http://127.0.0.1:5000
```

## EDA (Task 1.1)

```bash
python -m src.cli eda --config configs/eda.yaml --step N   # N = 1..10, see docs/eda_findings.md
```
