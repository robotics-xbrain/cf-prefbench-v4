from __future__ import annotations

from typing import Any


def impossible_premise_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    impossible = [r for r in rows if r.get("axis") == "impossible_premise"]
    if not impossible:
        return {
            "UCE": {"value": None, "available": False, "n": 0},
            "mean_confidence": {"value": None, "available": False, "n": 0},
            "forced_choice_bias": {"value": None, "available": False, "n": 0},
            "tie_rate": {"value": None, "available": False, "n": 0},
        }
    conf = [float(r["confidence"]) for r in impossible if r.get("confidence") is not None]
    ties = [r for r in impossible if r.get("prediction") == "Tie"]
    uce = conf
    return {
        "UCE": {"value": sum(uce) / len(uce) if uce else None, "available": bool(uce), "n": len(uce)},
        "mean_confidence": {"value": sum(conf) / len(conf) if conf else None, "available": bool(conf), "n": len(conf)},
        "forced_choice_bias": {"value": 1.0 - len(ties) / len(impossible), "available": True, "n": len(impossible)},
        "tie_rate": {"value": len(ties) / len(impossible), "available": True, "n": len(impossible)},
    }
