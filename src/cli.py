"""Command line entry point.

Usage:
    python -m src.cli eda --config configs/eda.yaml --step N
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


if __name__ == "__main__":
    main()
