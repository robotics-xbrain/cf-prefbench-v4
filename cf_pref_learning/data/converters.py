from __future__ import annotations

from pathlib import Path
from typing import Any

from .split import SPLITS
from ..utils.io import read_jsonl


def load_existing_cf_prefbench(project_root: str | Path) -> list[dict[str, Any]]:
    root = Path(project_root) / "data" / "cf_prefbench"
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        for row in read_jsonl(root / f"{split}.jsonl"):
            rows.append(row)
    return rows


def convert_sources_to_cf_prefbench(project_root: str | Path, sources: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Conservative converter.

    The current implementation only accepts already-normalized CF-PrefBench JSONL.
    External robotics datasets found during audit do not expose the full schema
    needed for scientifically valid preference examples, so they are reported as
    blockers rather than heuristically converted.
    """
    rows = load_existing_cf_prefbench(project_root)
    notes = []
    if not rows:
        notes.append("No existing data/cf_prefbench/*.jsonl files were found.")
    for source in sources:
        if source.get("can_support_cf_prefbench_construction"):
            notes.append(f"Potential source requires a custom, audited converter: {source.get('root')}")
    return rows, notes
