from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_csv_directory
from .modeling import train_and_evaluate


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a leakage-aware CIC-IDS2017 baseline.")
    parser.add_argument("data_directory", type=Path)
    parser.add_argument("--task", choices=("binary", "multiclass"), default="binary")
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)
    frame = load_csv_directory(args.data_directory, args.sample_fraction, args.random_state)
    result = train_and_evaluate(frame, args.task, random_state=args.random_state)
    print(
        json.dumps(
            {
                "task": result.task,
                "rows": len(frame),
                "balanced_accuracy": result.balanced_accuracy,
                "macro_f1": result.macro_f1,
                "labels": result.labels,
                "confusion_matrix": result.confusion_matrix.tolist(),
            },
            indent=2,
        )
    )

