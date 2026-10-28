# Experiment log

Every run that matters for the report, including the ones that failed. One entry per run:
date, who, what we tried, settings that differ from the config file, result, and why we think
it worked or failed. The run id links each entry to its full logs (W&B: `cub-dl-team/from-pixels-to-language/<id>`;
the first baseline was logged in MLflow before the switch).

| Date | Who | Run (name / id) | What we tried | Result (val) | Notes / why |
|---|---|---|---|---|---|
| 2026-10-28 | Sameh | `scratch-cnn-baseline` / `ce3cf43f9c7f41239e52ae86924b007c` | Scratch CNN baseline, `configs/cls_scratch.yaml` as committed (40 epochs, Kaggle T4, torch 2.11) | top-1: 43.3%, top-5: 73.5%, best epoch: 39 | Val loss levelled off at ~2.90 in the last 5 epochs without rising. Train top-1 61.9% vs val 43.3%: gap from ~epoch 20, expected with ~25 images/class. 13.1 min. Details: `docs/milestone1_sameh.md` |
| 2026-10-28 | Sameh | `scratch-cnn-baseline` / W&B `snt7by8w` | Same baseline re-run after the switch to W&B (same config and seed, commit `a9b1558`, Kaggle T4, torch 2.11) | top-1: 43.5%, top-5: 73.2%, best epoch: 37 (`evaluate`: 43.4% / 73.5%) | Matches the MLflow run within 0.2 points: cuDNN is not bit-deterministic (`train.deterministic: false`). 13.9 min. This is the run on the team dashboard. |
| 2026-10-28 | Adam | `resnet50-stage1-frozen` / W&B `xopc44on` | Transfer stage 1: pretrained ResNet50 (torchvision IMAGENET1K_V2) frozen, only the 200-class head trains, `configs/cls_transfer_stage1.yaml` as committed (10 epochs, batch 64, lr 1e-3; MacBook M3 Pro, MPS, torch 2.14) | top-1: 56.2%, top-5: 85.5%, best epoch: 8 | 0.41M of 23.9M weights trainable. Beats the scratch CNN by 13 points with no bird-specific learning in the encoder: ImageNet features alone separate many species but not look-alikes (warblers, sparrows). Flat after epoch 8 as the cosine lr reached ~0. Train 73.5% vs val 56.2%. 4.6 min. |
| 2026-10-28 | Adam | `resnet50-stage2-finetune` / W&B `hi8u8pjj` | Transfer stage 2: starts from the stage 1 `best.pt`, everything unfrozen, head lr 1e-3, encoder lr 1e-4 (`optim.backbone_lr`), `configs/cls_transfer_stage2.yaml` as committed (20 epochs, batch 32; MacBook M3 Pro, MPS) | top-1: 79.4%, top-5: 93.8%, best epoch: 17 | +23 points over stage 1, +36 over the scratch CNN. Overfits: train 99.8% vs val 79.4%. Val acc plateaus at 78–79% from epoch 10 and val loss flattens at ~1.70 from epoch 13 without rising, so the extra epochs neither help nor hurt. Next: tune weight decay, dropout and learning rates to close the gap. 29.7 min (~87 s/epoch). |
