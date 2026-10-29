# CLAUDE.md — From Pixels to Language (CSE485/CSE488 Deep Learning)

Context and rules for Claude Code in this repo. Read this before doing anything.

## Project in one paragraph

End-to-end multimodal deep learning project on CUB-200-2011 (bird photos + captions). One CNN **encoder** is the backbone for three tasks:

1. **Classification**: 200 bird species (Milestone 1)
2. **Colorization**: grayscale → color, with the encoder as the backbone of an encoder–decoder / U-Net (Milestone 2)
3. **Captioning**: frozen encoder → sequence decoder (LSTM/GRU/Transformer) that generates captions autoregressively (Milestone 2)

Course: CSE485/CSE488 Deep Learning, Ain Shams University Faculty of Engineering. Instructors: Dr. Mahmoud Khalil, Eng. Seif Yasser.

| Milestone  | Scope                                   | Deadline           |
|------------|-----------------------------------------|--------------------|
| M1         | EDA + classification + encoder training | 1 Nov 2026 (LMS)   |
| M2         | Colorization + captioning               | 18 Dec 2026        |
| Discussion | Individual, in person                   | from 19 Dec 2026   |

**Current focus: Milestone 1.** Don't start M2 code unless asked. Do design M1 code so M2 can plug into it.

**Status (29 Oct 2026):**
- Done: scratch CNN baseline (Sameh, 43.4% val), ResNet50 stage 1 (Adam, 56.2%) and stage 2 (Adam, 79.4%), grid + random search on stage 2 with the comparison (Karim, see [Hyperparameter tuning](#hyperparameter-tuning-required-in-m1)). Details: `docs/experiment_log.md`.
- Next: Adam retrains stage 2 for 20 epochs with the tuning winner (`grid-10`).
- Only after that final model exists: metrics (macro P/R/F1, confusion matrix, most-confused pairs), Grad-CAM, training curves, benchmark. Test-set numbers only for that final model.

## Team

- Team number: TeamXX <!-- TODO -->
- Members and GitHub handles: <!-- TODO -->

## How to work with us (read this first)

Grades are individual. Most of the grade comes from an in-person discussion (35%), where every member must explain the *whole* project, plus verifiable GitHub contributions. The course AI policy says we steer and you assist. So:

- **Plan before coding.** For any non-trivial task, propose the approach first:
  - which files you'll touch
  - the interfaces (inputs, outputs, tensor shapes)
  - the trade-offs

  Wait for a go-ahead before writing a lot of code.
- **Keep changes small and incremental.** One concern per change, no 600-line dumps. Each change should fit in one sensible commit.
- **Explain what you write.** After a change, say briefly what it does, why, and anything non-obvious (tensor shapes, why this loss or metric). Assume we'll be asked about it in the discussion.
- **Don't commit or push** unless explicitly asked. Suggest a commit message instead. Commits must come from the member who did the work.
- **Push back** if we ask for something that breaks the spec, leaks test data, or is a bad idea, and say why.
- **Flag claims we should verify.** If you state something about a paper, a published benchmark number, or a library API you're not sure of, say so. We'll check it.
- **Don't silently change** experiment settings, splits, seeds, or metrics. If something needs to change, say so first.

Workflow from the course: Think → Plan → Challenge → Refine → Define → Build incrementally → Review → Validate.

## Stack

- Python 3.10+, PyTorch + torchvision
- Hugging Face `datasets` for loading the data
- Matplotlib for plots (seaborn allowed alongside)
- Experiment tracking: **Weights & Biases** (W&B, inside CoreWeave Forge), one shared team project. Switched from MLflow on 28 Oct 2026 so all members' runs share one dashboard; use one tool only.
- Config: YAML files in `configs/`, loaded into dataclasses. No Hydra unless we decide otherwise.
- Pretrained models: torchvision or `timm`
- Metrics:
  - classification: scikit-learn
  - colorization: torchmetrics or our own code (PSNR/MAE)
  - captioning: BLEU/ROUGE via `nltk` / `rouge-score` or HF `evaluate`
- Compute: free Colab / Kaggle T4 (16 GB VRAM) or local machines. Everything must fit that.

## Repo structure

```
project/
├── CLAUDE.md
├── README.md              # setup + exact train/inference commands (graded)
├── requirements.txt
├── configs/               # one YAML per experiment/task
│   ├── cls_scratch.yaml
│   ├── cls_transfer.yaml
│   └── tuning/            # search spaces for grid/random search
├── data/splits/           # committed: validation split indices (small JSON)
├── docs/
│   └── experiment_log.md  # what we tried, what failed, why (feeds the report)
├── notebooks/             # experiments only, never imported by src/
├── src/
│   ├── data/              # datasets, transforms, augmentation, caption cleaning, vocab
│   ├── models/
│   │   ├── encoders/      # scratch CNN + pretrained wrappers, all sharing one interface
│   │   ├── heads/         # classification head (later: colorization decoder, caption decoder)
│   │   └── registry.py    # name → constructor, so configs pick models by string
│   ├── training/          # trainer, loops, checkpointing
│   ├── tuning/            # grid search, random search, comparison
│   ├── evaluation/        # metrics, benchmarking
│   ├── visualization/     # plots, Grad-CAM
│   ├── utils/             # seeding, logging, config loading, device
│   └── cli.py             # single entry point
├── tests/                 # quick shape/sanity tests
└── outputs/               # figures, logs, tuning/<search>/results.csv, checkpoints (checkpoints gitignored)
```

Keep it this simple. Don't over-engineer.

## Commands

These are the target commands. Keep the CLI matching them and keep README.md in sync.

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# EDA (writes plots to outputs/figures/eda/)
python -m src.cli eda --config configs/eda.yaml

# Classification
python -m src.cli train    --task classification --config configs/cls_scratch.yaml
python -m src.cli train    --task classification --config configs/cls_transfer.yaml
python -m src.cli tune     --task classification --method grid   --config configs/tuning/cls_grid.yaml \
                           --override model.init_from=outputs/checkpoints/<stage1_run>/best.pt
python -m src.cli tune     --task classification --method random --config configs/tuning/cls_random.yaml \
                           --override model.init_from=outputs/checkpoints/<stage1_run>/best.pt
python -m src.cli compare-tuning --configs configs/tuning/cls_grid.yaml configs/tuning/cls_random.yaml
python -m src.cli evaluate --task classification --checkpoint outputs/checkpoints/<run>.pt
python -m src.cli predict  --task classification --image path/to/bird.jpg --checkpoint outputs/checkpoints/<run>.pt
python -m src.cli gradcam  --checkpoint outputs/checkpoints/<run>.pt --image path/to/bird.jpg

# M2 (later)
python -m src.cli train   --task colorization --config configs/color.yaml
python -m src.cli predict --task colorization --image path/to/gray.jpg --checkpoint ...
python -m src.cli train   --task caption --config configs/caption.yaml
python -m src.cli predict --task caption --image path/to/bird.jpg --checkpoint ...

# Tracking (once per machine; on Kaggle use the WANDB_API_KEY secret)
wandb login

# Tests
pytest -q
```

## Data

Dataset: CUB-200-2011 with captions (Reed et al., 2016), from Hugging Face:

```python
from datasets import load_dataset
train = load_dataset("Multimodal-Fatima/CUB_train", split="train")  # 5,994 images
test  = load_dataset("Multimodal-Fatima/CUB_test",  split="test")   # 5,794 images
```

Fields:

| Field         | Content                                                        |
|---------------|----------------------------------------------------------------|
| `image`       | PIL image                                                      |
| `label`       | 200-class species label                                        |
| `description` | 10 captions in one string, separated by `\n`                   |
| `file_name`   | original file name                                             |
| `id`          | row identifier                                                 |

Read label names from `train.features` instead of hardcoding them.

### Data rules

- **Splits.**
  - Use the official train/test split only.
  - Validation = a stratified ~10–15% of train, with a fixed seed. Save the indices to `data/splits/` and commit them so the whole team uses the same split.
  - **The test set is used only for final evaluation and benchmarking.** Never use it for tuning, early stopping, or model selection.
- **Color mode.** Always call `.convert("RGB")` when loading. Some CUB images may be grayscale or in other modes; the EDA should count them.
- **Captions.** The pipeline is: split `description` on `\n` → strip → drop empty strings → lowercase → normalize whitespace and punctuation. Check that each image has 10 captions and log any that don't. Every cleaning step must be justified in the report, so keep the steps explicit and documented (not one magic regex).
- **Vocabulary (for M2).**
  - Build it from **training captions only**.
  - Special tokens: `<pad> <bos> <eos> <unk>`.
  - Choose the min-frequency threshold from the EDA word-frequency plot.
- **Augmentation.**
  - Train only: RandomResizedCrop, horizontal flip, and small rotations.
  - **No vertical flips.**
  - **Keep color jitter mild or off.** Color is a key cue for telling species apart.
  - Val/test: resize + center crop only.
- **Normalization.** ImageNet mean/std for pretrained models. For the scratch model, use either the same or dataset stats. Pick one and document it.
- **Image size.** 224×224 by default, set in config.
- **Caching.** Cache the HF dataset (`cache_dir`, or mount Drive on Colab). Don't re-download every run.
- **CUB extras.** The original release's bounding boxes, parts, and attributes aren't used unless we decide otherwise.

## Architecture contract (important for M2)

The encoder is shared by all three tasks, so it's its own module with a fixed interface. Every encoder (scratch or pretrained) implements:

```python
@dataclass
class EncoderOutput:
    features: list[Tensor]  # multi-scale feature maps, shallow → deep (U-Net skips, caption attention)
    pooled: Tensor          # (B, out_dim) global-average-pooled vector (classification)

class Encoder(nn.Module):
    in_channels: int           # 3 for RGB; colorization may use 1
    stage_channels: list[int]  # channels of each returned feature map
    out_dim: int               # size of the pooled vector
    gradcam_layer: nn.Module   # last conv block, used by Grad-CAM

    def forward(self, x: Tensor) -> EncoderOutput: ...
    def freeze(self) -> None: ...
    def unfreeze(self, stages: list[int] | None = None) -> None: ...
```

- **Task model = encoder + head**, e.g. `Classifier(encoder, ClassificationHead)`. Heads never reach inside the encoder.
- **Models are built from config** through `registry.py`. Swapping encoders must not require touching the trainer.
- **Grayscale input for colorization (M2)** is handled by a documented option: either repeat gray → 3 channels, or adapt the first conv. Don't hardcode 3 channels deep inside the encoder.
- **`freeze` / `unfreeze`** are used for transfer learning now and for the frozen-encoder captioning task later.

## Code conventions

- **SOLID, without over-engineering.** Prefer simple classes and functions. No abstract base class with a single implementation, except the Encoder interface above.
- **Types and docstrings.** Type hints on public functions. Short docstrings that state tensor shapes, e.g. `# (B, C, H, W)`.
- **No hardcoded values in `src/`.** Paths, hyperparameters, and magic numbers come from config.
- **Notebooks stay separate.** `src/` never imports from `notebooks/`. If a notebook proves something useful, move it into `src/`.
- **Device-agnostic.** Use `cuda` if available, else `cpu` (`mps` is fine locally).
- **Logging.** Use `logging` in library code, not stray `print`s. The CLI can print summaries.
- **Tests.** Caption cleaning, transforms, metrics, and model output shapes get quick pytest tests.
- **Lint/format.** ruff (optional but preferred).

## Training

- **One `Trainer`** works for any task, given model, loss, optimizer, metrics, and loaders. Task-specific logic lives in small task classes or functions, not in `if task == ...` chains inside the loop.
- **Mixed precision** (`torch.autocast` + GradScaler) is on by default for the T4.
- **Checkpoints.** Save both best (by val metric) and last. Each checkpoint stores model state, optimizer state, epoch, config, and val metric, so a run can resume after a Colab disconnect.
- **Early stopping** uses the validation set, never test.
- **The scratch baseline is fully from scratch.** Random init, no pretrained weights anywhere.
- **Transfer learning.** Record what was frozen vs unfrozen and the learning rates for backbone vs head.
- **Local Mac (MPS) runs** (Karim trains on an M4, 16 GB):
  - AMP is CUDA-only, so MPS runs in fp32. ResNet50 stage 2 takes ~110 s/epoch at batch 32.
  - Long runs: start with `caffeinate -i ...`, keep the charger in and the lid open. Closing the lid on battery sleeps the Mac anyway. A low battery slows training.
  - `CERTIFICATE_VERIFY_FAILED` when torch.hub downloads weights (python.org Python): run `/Applications/Python 3.13/Install Certificates.command`, or prefix the command with `SSL_CERT_FILE="$(python -m certifi)"`.

## Reproducibility

- `set_seed(seed)` seeds python, numpy, and torch (CPU + CUDA). The seed comes from config and is logged.
- DataLoader workers are seeded, and a generator is passed for shuffling.
- Every run logs its full config, git commit hash, seed, and library versions.
- `requirements.txt` stays current: update it whenever a package is added.
- Someone else must be able to reproduce our results from README.md + requirements.txt alone. This is graded.

## Experiment tracking

- **Every training run is tracked.** No untracked "quick runs" whose results end up in the report.
- **Log per run:**
  - all hyperparameters (flattened config)
  - per-epoch train/val loss and metrics, plus learning rate and epoch time
  - final test metrics (final models only)
  - checkpoint path
  - key figures as artifacts
- **Experiment names** (W&B group + tag): `cls-scratch`, `cls-transfer`, `cls-tuning-grid`, `cls-tuning-random` (later `colorization`, `captioning`).
- **Run names** say what changed, e.g. `resnet50-ft-lr3e-4-bs32`.
- **Tuning trials** share one W&B `group` per search (`train_classification(cfg, group=...)`).
- `wandb/` (local run files) is gitignored. API keys are never committed. Screenshots or exports go into the report.
- **Offline mode** for long local runs: `--override tracking.wandb_mode=offline`, then `wandb sync wandb/offline-run-*` afterwards. The overnight tuning ran this way because an online upload once hung and crashed a finished trial.
- **File uploads are best effort** (`WandbTracker.log_artifact`): a failed upload logs a warning and the run goes on; the file stays on disk.

## Hyperparameter tuning (required in M1)

- Tune a **subset** of hyperparameters, e.g. learning rate, weight decay, dropout, batch size.
- Run **both Grid Search and Random Search**, with their search spaces in `configs/tuning/`.
- Give both methods a comparable budget (similar trials × epochs) so the comparison is fair.
- **Compare them on:** best val score, number of trials, and total wall-clock time. Produce a table and a plot.
- To fit Colab limits, tune with short runs (few epochs or a data subset), then fully retrain the best config.
- Optuna is optional, and only after the two required methods are done.

### How it is implemented

- `src/tuning/search.py` makes the trials. Each trial is a dict of `key=value` overrides, the same format as `cli train --override`.
  - Grid: `itertools.product`, where the last key changes fastest.
  - Random: a seeded numpy generator (`sampler_seed`). Learning rates and weight decay are drawn log-uniformly, and values are rounded to 3 significant digits.
- `src/tuning/run.py` runs the trials. Each trial is an ordinary `train_classification` run in W&B group = the experiment name.
  - The dataset is loaded once for all trials.
  - One row is appended to `outputs/tuning/<experiment>/results.csv` after every trial.
  - Rerunning the command skips trials with status `ok` and retries failed ones. Failed rows are kept as a record.
  - Trial `.pt` files are deleted after each trial (`keep_checkpoints: false`); `history.csv` is kept.
- `src/tuning/compare.py` + `src/visualization/tuning.py` (`cli compare-tuning`) build the table and two figures in `outputs/figures/tuning/`.
  - Only successful trials count toward the time. Failed attempts are listed separately.
  - `hours_to_best` is the time until a search found its best trial.

### Results (29 Oct 2026, done)

- **Setup:** stage 2, 12 trials per method, 8 epochs per trial. Every trial starts from the stage 1 `best.pt` (`xopc44on`) with seed 42, on Karim's Mac.
- **Spaces:**
  - Grid: backbone lr {3e-5, 1e-4, 3e-4} × weight decay {5e-4, 5e-3} × dropout {0.3, 0.5}.
  - Random: the same three settings plus the head lr, drawn from ranges.
- **Winner: `grid-10`** (W&B `6okn99lk`), 79.3% val top-1. Settings: `optim.backbone_lr=3e-4`, `optim.weight_decay=5e-4`, `model.dropout=0.5`, `optim.lr=1e-3`.
- **Best random trial:** `random-05` (W&B `ywghkztm`), 78.9%.
- **Time:** both searches took ~3.4 h. Random found its best after 1.5 h, grid after 2.8 h.
- **Only the backbone lr matters:** 3e-5 gives ~72%, 1e-4 ~75–76%, 3e-4 ~78–79%. Weight decay, dropout and head lr changed results by ≤ ~1 point.
- **Caveat:** 3e-4 is the top of the grid, and its trials were still improving at epoch 8.
- **Open TODO:** verify the Bergstra & Bengio (2012) citation before it goes into the report.

## Evaluation

| Task               | Metrics                                                    | Notes |
|--------------------|------------------------------------------------------------|-------|
| Classification     | top-1 accuracy, top-5 accuracy, macro precision/recall/F1, confusion matrix | With 200 classes, show the **top-N most-confused species pairs**. The full matrix can go in an appendix. |
| Colorization (M2)  | PSNR, MAE (SSIM optional)                                   | |
| Captioning (M2)    | BLEU-1..4, ROUGE-L (METEOR/CIDEr optional)                  | Score against all 10 reference captions. |

- For every metric, the report must explain what it measures and what our score means. Keep a short note in the `src/evaluation/` docstrings so it's easy to pull into the report.
- **Benchmarking:**
  - Compare against open-source models or published CUB results, using the **same official test split and the same metrics**.
  - Explain where the differences come from. Scoring lower is fine.
  - **Never invent or guess published numbers.** Cite the source, or leave a TODO.

## Grad-CAM

- Implement it ourselves with forward/backward hooks on `encoder.gradcam_layer`, not a black-box library call. We have to be able to explain it.
- Produce heatmaps for **correct and wrong** predictions, for both the scratch and the transfer model.
- Save grids showing: original | heatmap overlay | predicted vs true label.
- The analysis should answer: is the model looking at the bird, or at the background (water, branches, sky)?

## Required visualizations

- **Data:** sample images, class distribution, image size/aspect stats, caption length and vocab stats, augmentation examples.
- **Training:** train vs val loss and metric curves for every reported model.
- **Comparison:** scratch vs transfer, architecture experiments, tuning results, the final selected model.
- **Explainability:** Grad-CAM grids.
- **M2:** grayscale | colorized | ground truth grids; generated vs reference captions.

All figures are generated by code into `outputs/figures/<task>/`, with titles and labeled axes. No figure should exist only inside a notebook.

## Milestone 1 checklist

- [ ] Repo, structure, requirements.txt, README skeleton
- [ ] Dataset loading, RGB conversion, stratified val split saved
- [ ] EDA: classes, images, captions, anomalies, with plots saved
- [ ] Caption cleaning pipeline + stats
- [ ] Augmentation pipeline + example grid
- [ ] Encoder interface + scratch CNN
- [ ] Trainer + W&B logging + checkpointing
- [ ] Scratch baseline trained and evaluated
- [ ] Transfer-learning model trained and evaluated
- [x] Grid search + random search + comparison (Karim, 29 Oct)
- [ ] Grad-CAM (correct + wrong, both models)
- [ ] CLI: train / evaluate / predict / gradcam for classification
- [ ] README commands verified from a clean environment
- [ ] Submit to LMS by 1 Nov

## Milestone 2 (later; don't start unless asked)

- **Colorization:** encoder–decoder or U-Net with the trained encoder as the backbone. Build grayscale/color pairs on the fly. Consider predicting in Lab space (L in → ab out); decide and justify.
- **Captioning:** **freeze** the encoder and project its features into the decoder (LSTM/GRU/Transformer). Use teacher forcing in training and greedy/beam search at inference.
- Both tasks need CLI train + predict. Tuning is optional in M2.

## Git

- **Never commit:** datasets, checkpoints (`*.pt`, `*.pth`, `*.ckpt`), `mlruns/`, `wandb/`, large outputs, `.env`, API keys. Keep `.gitignore` up to date.
- Trained models are shared through Google Drive (a final deliverable), not git.
- Work on feature branches and merge into `main`. `main` must always run.
- Commit messages are imperative and specific, e.g. `Add stratified val split with saved indices`.
- Don't rewrite shared history (no force-push to `main`).
- Everyone commits from their own account, regularly. Uploading at the deadline is explicitly penalized.

## Final deliverables (TeamXX.zip on LMS)

- Google Drive link to all trained models
- GitHub link with full commit history
- README.md with setup and exact commands
- requirements.txt
- Report PDF
- Video demo of the end-to-end system

The report must document everything, **including what didn't work**. When an experiment fails, add an entry to `docs/experiment_log.md`: date, what we tried, the result, and why we think it failed.

## Don't

- Use the test set for anything except final evaluation.
- Put final logic in notebooks.
- Add a dependency without updating requirements.txt.
- Make large multi-concern changes in one go.
- Fabricate metrics, benchmark numbers, or citations.
- Start Milestone 2 work while Milestone 1 is open, unless asked.
