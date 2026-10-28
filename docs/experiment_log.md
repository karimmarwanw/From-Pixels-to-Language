# Experiment log

Every run that matters for the report, including the ones that failed. One entry per run:
date, who, what we tried, settings that differ from the config file, result, and why we think
it worked or failed. The MLflow run id links each entry to its full logs.

| Date | Who | Run (MLflow name / id) | What we tried | Result (val) | Notes / why |
|---|---|---|---|---|---|
| 2026-10-28 | Sameh | `scratch-cnn-baseline` / `ce3cf43f9c7f41239e52ae86924b007c` | Scratch CNN baseline, `configs/cls_scratch.yaml` as committed (40 epochs, Kaggle T4, torch 2.11) | top-1: 43.3%, top-5: 73.5%, best epoch: 39 | Val loss levelled off at ~2.90 in the last 5 epochs without rising. Train top-1 61.9% vs val 43.3%: gap from ~epoch 20, expected with ~25 images/class. 13.1 min. Details: `docs/milestone1_sameh.md` |
