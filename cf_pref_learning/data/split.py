from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

SPLITS = (
    "train",
    "val",
    "test_seen",
    "test_heldout_lexical",
    "test_heldout_camera",
    "test_heldout_color",
    "test_heldout_spatial",
    "test_hard_negatives",
)


def split_existing_or_assign(rows: list[dict[str, Any]], seed: int = 42) -> list[dict[str, Any]]:
    if not rows:
        return rows
    if all(row.get("split") in SPLITS for row in rows):
        return rows
    rng = random.Random(seed)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row.get("counterfactual_group_id", row.get("pair_id", row.get("example_id"))))].append(row)
    groups = sorted(by_group)
    rng.shuffle(groups)
    n = len(groups)
    heldout = set(groups[int(0.8 * n):])
    val = set(groups[int(0.7 * n):int(0.8 * n)])
    test_seen = set(groups[int(0.6 * n):int(0.7 * n)])
    for group in groups:
        split = "train"
        if group in heldout:
            split = "test_heldout_lexical"
        elif group in val:
            split = "val"
        elif group in test_seen:
            split = "test_seen"
        for row in by_group[group]:
            row["split"] = split
    return rows
