from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GPUStatus:
    index: int
    memory_used_mb: int
    utilization_gpu: int


def query_gpus() -> list[GPUStatus]:
    cmd = ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return []
    statuses = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            statuses.append(GPUStatus(index=int(parts[0]), memory_used_mb=int(parts[1]), utilization_gpu=int(parts[2])))
    return statuses


def select_idle_gpus(max_gpus: int = 3, reserve_gpus: int = 1, memory_threshold_mb: int = 1500, util_threshold: int = 20) -> list[int]:
    statuses = query_gpus()
    idle = [g.index for g in statuses if g.memory_used_mb <= memory_threshold_mb and g.utilization_gpu <= util_threshold]
    usable = max(0, min(max_gpus, len(statuses) - reserve_gpus))
    return idle[:usable]

