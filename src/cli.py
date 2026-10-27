"""Command line entry point.

Usage:
    python -m src.cli eda --config configs/eda.yaml --step N
"""

import argparse
import logging


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.cli", description="From Pixels to Language")
    sub = parser.add_subparsers(dest="command", required=True)

    eda = sub.add_parser("eda", help="Run one step of the exploratory data analysis (Task 1.1)")
    eda.add_argument("--config", required=True, help="Path to the EDA YAML config")
    eda.add_argument("--step", type=int, required=True, help="EDA step number to run")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2"):  # the HF hub logs one line per HTTP request
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.command == "eda":
        from src.eda_steps import STEPS
        from src.utils.config import load_eda_config

        if args.step not in STEPS:
            parser.error(f"--step must be one of {sorted(STEPS)}")
        STEPS[args.step](load_eda_config(args.config))


if __name__ == "__main__":
    main()
