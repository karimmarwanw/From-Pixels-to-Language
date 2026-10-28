# Experiment log

Every run that matters for the report, including the ones that failed. One entry per run:
date, who, what we tried, settings that differ from the config file, result, and why we think
it worked or failed. The run id links each entry to its full logs (W&B: `cub-dl-team/from-pixels-to-language/<id>`;
the first baseline was logged in MLflow before the switch).

| Date | Who | Run (name / id) | What we tried | Result (val) | Notes / why |
|---|---|---|---|---|---|
| 2026-10-28 | Sameh | `scratch-cnn-baseline` / `ce3cf43f9c7f41239e52ae86924b007c` | Scratch CNN baseline, `configs/cls_scratch.yaml` as committed (40 epochs, Kaggle T4, torch 2.11) | top-1: 43.3%, top-5: 73.5%, best epoch: 39 | Val loss levelled off at ~2.90 in the last 5 epochs without rising. Train top-1 61.9% vs val 43.3%: gap from ~epoch 20, expected with ~25 images/class. 13.1 min. Details: `docs/milestone1_sameh.md` |
| 2026-10-28 | Sameh | `scratch-cnn-baseline` / W&B `snt7by8w` | Same baseline re-run after the switch to W&B (same config and seed, commit `a9b1558`, Kaggle T4, torch 2.11) | top-1: 43.5%, top-5: 73.2%, best epoch: 37 (`evaluate`: 43.4% / 73.5%) | Matches the MLflow run within 0.2 points: cuDNN is not bit-deterministic (`train.deterministic: false`). 13.9 min. This is the run on the team dashboard. |
