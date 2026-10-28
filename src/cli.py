"""Command line entry point.

Usage:
    python -m src.cli eda      --config configs/eda.yaml --step N
    python -m src.cli train    --task classification --config configs/cls_scratch.yaml [--override optim.lr=3e-4 ...]
    python -m src.cli train    --task classification --config configs/cls_scratch.yaml --resume <run_dir>/last.pt
    python -m src.cli tune     --task classification --method grid --config configs/tuning/cls_grid.yaml \
                               --override model.init_from=<stage1_run_dir>/best.pt [--max-trials 1]
    python -m src.cli evaluate --task classification --checkpoint <run_dir>/best.pt [--split val|test]
    python -m src.cli predict  --task classification --checkpoint <run_dir>/best.pt --image bird.jpg [--topk 5]
"""

import argparse
import contextlib
import logging
import sys
from pathlib import Path
from typing import TextIO


class _Tee:
    """Write everything to several streams at once (here: the terminal and a text file)."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> None:
        for s in self.streams:
            s.write(text)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.cli", description="From Pixels to Language")
    sub = parser.add_subparsers(dest="command", required=True)

    eda = sub.add_parser("eda", help="Run one step of the exploratory data analysis (Task 1.1)")
    eda.add_argument("--config", required=True, help="Path to the EDA YAML config")
    eda.add_argument("--step", type=int, required=True, help="EDA step number to run")

    train = sub.add_parser("train", help="Train a model from a YAML config, tracked in W&B")
    train.add_argument("--task", choices=["classification"], default="classification")
    train.add_argument("--config", required=True, help="Path to the training YAML config")
    train.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                       help="Change config values, e.g. optim.lr=3e-4 train.epochs=5")
    train.add_argument("--resume", help="last.pt of an interrupted run, to continue it")

    tune = sub.add_parser("tune", help="Grid or random search: one W&B-tracked training run per trial")
    tune.add_argument("--task", choices=["classification"], default="classification")
    tune.add_argument("--method", choices=["grid", "random"], required=True)
    tune.add_argument("--config", required=True, help="Tuning YAML in configs/tuning/")
    tune.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                      help="Training config changes for EVERY trial, e.g. model.init_from=<stage1>/best.pt")
    tune.add_argument("--max-trials", type=int, help="Run only the first N trials (1 = test the setup)")
    tune.add_argument("--results-dir", help="Where results.csv goes (default: results_dir in the config)")

    evaluate = sub.add_parser("evaluate", help="Evaluate a checkpoint on the val set (or the test set, final only)")
    evaluate.add_argument("--task", choices=["classification"], default="classification")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--split", choices=["val", "test"], default="val")
    evaluate.add_argument("--out-dir", default="outputs/predictions", help="Where to save logits + labels (.npz)")

    predict = sub.add_parser("predict", help="Predict the species of one image")
    predict.add_argument("--task", choices=["classification"], default="classification")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--image", required=True)
    predict.add_argument("--topk", type=int, default=5)

    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # captions contain characters the Windows console codepage cannot print
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2"):  # the HF hub logs one line per HTTP request
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.command == "eda":
        from src.eda_steps import STEPS
        from src.utils.config import load_eda_config

        if args.step not in STEPS:
            parser.error(f"--step must be one of {sorted(STEPS)}")
        cfg = load_eda_config(args.config)
        # Print to the terminal AND save the same text to outputs/eda/stepNN_output.txt
        out_txt = Path(cfg.stats_dir) / f"step{args.step:02d}_output.txt"
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        with open(out_txt, "w", encoding="utf-8") as f, contextlib.redirect_stdout(_Tee(sys.stdout, f)):
            STEPS[args.step](cfg)
        print(f"\nThis output was also saved to {out_txt}")

    elif args.command == "train":
        from src.training.run import train_classification
        from src.utils.config import load_train_config

        cfg = load_train_config(args.config, args.override)
        if cfg.task != args.task:
            parser.error(f"--task {args.task} but the config says task: {cfg.task}")
        result = train_classification(cfg, resume=args.resume)
        print(f"\nBest {cfg.train.monitor}: {result['best_metric']:.4f} (epoch {result['best_epoch']}, "
              f"{result['epochs_run']} epochs, {result['train_time_s'] / 60:.1f} min)")
        print(f"Checkpoints: {result['ckpt_dir']}/best.pt and last.pt")
        print(f"W&B run:     {result['run_id']}  (project {cfg.tracking.wandb_project})")

    elif args.command == "tune":
        from src.tuning.run import run_search, summarize
        from src.utils.config import load_train_config, load_tune_config

        cfg = load_tune_config(args.config)
        if cfg.method != args.method:
            parser.error(f"--method {args.method} but the config says method: {cfg.method}")
        base = load_train_config(cfg.base_config, args.override)
        if base.task != args.task:
            parser.error(f"--task {args.task} but {cfg.base_config} says task: {base.task}")
        if not base.model.init_from:
            print("WARNING: no model.init_from: trials start from ImageNet weights + a random head, "
                  "not from the stage 1 checkpoint. Fine for a smoke test only.")
        out = run_search(cfg, args.override, args.max_trials, args.results_dir)

        metric = f"best_{base.train.monitor}"
        s = summarize(out, metric, base.train.monitor_mode)
        print(f"\n{cfg.method} search {cfg.experiment}: {s['n_ok']} ok, {s['n_failed']} failed, "
              f"{s['total_time_s'] / 60:.1f} min in total")
        if s["best"]:
            b = s["best"]
            print(f"Best trial: {b['run_name']}  {metric} {float(b[metric]):.4f}  (W&B run {b['run_id']})")
            print("Settings:   " + "  ".join(f"{k}={b[k]}" for k in cfg.space))
        print(f"All trials: {out}")

    elif args.command == "evaluate":
        from src.evaluation.inference import evaluate_checkpoint

        r = evaluate_checkpoint(args.checkpoint, args.split, args.out_dir)
        print(f"\n{args.split} set ({r['n']} images): top-1 acc {r['acc']:.4f}  top-5 acc {r['top5']:.4f}  "
              f"loss {r['loss']:.4f}")
        print(f"Predictions saved to {r['predictions_file']}")

    elif args.command == "predict":
        from src.evaluation.inference import predict_image

        print(f"\nTop-{args.topk} predictions for {args.image}:")
        for name, p in predict_image(args.checkpoint, args.image, args.topk):
            print(f"  {p:6.1%}  {name}")


if __name__ == "__main__":
    main()
