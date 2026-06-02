from __future__ import annotations

from typing import Any


def unavailable_baselines(reason: str) -> list[dict[str, Any]]:
    return [
        {"name": name, "available": False, "reason": reason}
        for name in ["language_only", "shuffled_video", "clip_score", "r3m_head", "vip_head", "liv", "roboclip", "vlp", "base_pref"]
    ]

