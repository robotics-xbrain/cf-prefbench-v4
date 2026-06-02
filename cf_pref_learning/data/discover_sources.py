from __future__ import annotations

from collections import defaultdict
import os
from pathlib import Path
from typing import Any

KEYWORDS = [
    "vlp", "mtvlp", "metaworld", "maniskill", "trajectory", "traj", "preference", "pref",
    "robot", "video", "clip", "liv", "roboclip", "vlmrm", "r3m", "vip",
]
DATA_EXTS = {".json", ".jsonl", ".csv", ".pkl", ".hdf5", ".h5", ".npz", ".npy", ".mp4", ".avi"}
VIDEO_EXTS = {".mp4", ".avi"}
META_EXTS = {".json", ".jsonl", ".csv"}
TRAJ_EXTS = {".hdf5", ".h5", ".npz", ".npy", ".pkl"}


def _skip(path: Path) -> bool:
    parts = set(path.parts)
    return bool(parts & {"conda", "envs", ".git", "__pycache__", "_tools"})


def discover_project_sources(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    files: list[Path] = []
    for path in _walk_files(root, max_files=10000):
        if path.suffix.lower() in DATA_EXTS:
            files.append(path)
    return summarize_files(root, files, inside_project=True)


def summarize_external_paths(paths: list[str | Path], max_files_per_root: int = 2000) -> list[dict[str, Any]]:
    summaries = []
    for raw in paths:
        path = Path(raw)
        files = []
        if path.exists():
            iterator = _walk_files(path, max_files=max_files_per_root) if path.is_dir() else [path]
            for item in iterator:
                if item.suffix.lower() in DATA_EXTS:
                    files.append(item)
                if len(files) >= max_files_per_root:
                    break
        summaries.append(summarize_files(path, files, inside_project=False))
    return summaries


def summarize_files(root: Path, files: list[Path], inside_project: bool) -> dict[str, Any]:
    by_ext: dict[str, int] = defaultdict(int)
    keyword_hits: dict[str, int] = defaultdict(int)
    for file in files:
        by_ext[file.suffix.lower()] += 1
        name = str(file).lower()
        for keyword in KEYWORDS:
            if keyword in name:
                keyword_hits[keyword] += 1
    has_video = any(p.suffix.lower() in VIDEO_EXTS for p in files)
    has_trajectory = any(p.suffix.lower() in TRAJ_EXTS for p in files)
    has_language = any("instruction" in str(p).lower() or "language" in str(p).lower() for p in files)
    has_preference = any("pref" in str(p).lower() or "preference" in str(p).lower() for p in files)
    has_task_meta = any(p.suffix.lower() in META_EXTS for p in files)
    return {
        "root": str(root),
        "exists": root.exists(),
        "inside_project": inside_project,
        "file_count": len(files),
        "counts_by_extension": dict(sorted(by_ext.items())),
        "keyword_hits": dict(sorted(keyword_hits.items())),
        "sample_files": [str(p) for p in sorted(files)[:80]],
        "contains_trajectory_or_array_files": has_trajectory,
        "contains_video_files": has_video,
        "contains_language_instructions_detected": has_language,
        "contains_preference_labels_detected": has_preference,
        "contains_task_metadata_detected": has_task_meta,
        "can_support_cf_prefbench_construction": bool(has_video and has_language and has_preference),
        "missing_for_cf_prefbench": [
            item for item, present in [
                ("trajectory/video pairs", has_video),
                ("language instructions", has_language),
                ("preference labels", has_preference),
                ("task metadata", has_task_meta),
            ]
            if not present
        ],
    }


def _walk_files(root: Path, max_files: int) -> list[Path]:
    if not root.exists():
        return []
    out: list[Path] = []
    skip_names = {"conda", "envs", ".git", "__pycache__", "_tools", ".cache"}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_names]
        for filename in filenames:
            path = Path(dirpath) / filename
            if _skip(path):
                continue
            out.append(path)
            if len(out) >= max_files:
                return out
    return out
