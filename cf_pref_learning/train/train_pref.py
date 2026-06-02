from __future__ import annotations

import argparse
from pathlib import Path

from ..utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    write_json(Path(args.output_dir) / "status.json", {"status": "blocked", "reason": "No real CF-PrefBench training data is available."})


if __name__ == "__main__":
    main()

