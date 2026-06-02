from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .io import append_jsonl, ensure_dir, write_json
from .logging import utc_now


@dataclass
class Job:
    job_id: str
    command: list[str]
    stage: str
    gpu: int | None = None
    attempts: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class JobQueue:
    def __init__(self, project_root: str | Path):
        self.root = Path(project_root)
        ensure_dir(self.root / "outputs" / "autopilot")
        self.registry = self.root / "outputs" / "autopilot" / "job_registry.jsonl"

    def record(self, job: Job, status: str, returncode: int | None = None) -> None:
        append_jsonl(
            self.registry,
            {
                "timestamp_utc": utc_now(),
                "job_id": job.job_id,
                "stage": job.stage,
                "command": job.command,
                "gpu": job.gpu,
                "attempts": job.attempts,
                "status": status,
                "returncode": returncode,
                "metadata": job.metadata,
            },
        )

    def heartbeat(self, stage: str, status: str) -> None:
        write_json(self.root / "outputs" / "autopilot" / "heartbeat.json", {"timestamp_utc": utc_now(), "stage": stage, "status": status})
        write_json(self.root / "outputs" / "autopilot" / "current_stage.json", {"stage": stage, "status": status})

    def run_job(self, job: Job, max_retries: int = 2) -> int:
        while job.attempts <= max_retries:
            job.attempts += 1
            self.record(job, "started")
            proc = subprocess.run(job.command, cwd=self.root, check=False)
            self.record(job, "completed" if proc.returncode == 0 else "failed", proc.returncode)
            if proc.returncode == 0:
                return 0
        return proc.returncode

