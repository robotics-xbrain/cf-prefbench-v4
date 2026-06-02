from __future__ import annotations

import argparse
from pathlib import Path

from .metrics import aggregate_metrics
from ..utils.io import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.predictions)
    write_json(Path(args.output), aggregate_metrics(rows))


if __name__ == "__main__":
    main()

