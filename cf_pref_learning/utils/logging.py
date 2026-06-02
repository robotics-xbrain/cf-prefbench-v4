from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import append_jsonl


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_status(project_root: str | Path) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["git", "status", "--short"],
            cwd=project_root,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except Exception as exc:  # pragma: no cover - defensive provenance path
        return {"available": False, "error": str(exc)}
    return {
        "available": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def log_command(project_root: str | Path, name: str, argv: list[str], config: dict[str, Any]) -> None:
    append_jsonl(
        Path(project_root) / "outputs" / "command_logs" / f"{name}.jsonl",
        {
            "timestamp_utc": utc_now(),
            "argv": argv,
            "cwd": str(Path(project_root).resolve()),
            "python": sys.version,
            "platform": platform.platform(),
            "config": config,
            "git_status": git_status(project_root),
        },
    )

