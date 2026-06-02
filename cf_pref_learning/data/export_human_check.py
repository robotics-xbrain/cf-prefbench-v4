from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from ..utils.io import write_jsonl


def export_human_check(rows: list[dict[str, Any]], out_path: str | Path, seed: int = 42, min_n: int = 200, max_n: int = 500) -> dict[str, Any]:
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    n = min(max_n, len(rows))
    sample = rows[:n]
    write_jsonl(out_path, sample)
    return {
        "requested_min": min_n,
        "requested_max": max_n,
        "exported": n,
        "limitation": None if n >= min_n else "Fewer than 200 viable examples are available.",
    }

