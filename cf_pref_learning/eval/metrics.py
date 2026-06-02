from __future__ import annotations

from collections import defaultdict
from typing import Any


def _metric(value: float | None, n: int) -> dict[str, Any]:
    return {"value": value, "available": value is not None, "n": n}


def standard_accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eval_rows = [r for r in rows if r.get("preferred") != "Tie" and r.get("prediction") in {"A", "B"}]
    if not eval_rows:
        return _metric(None, 0)
    return _metric(sum(r["prediction"] == r["preferred"] for r in eval_rows) / len(eval_rows), len(eval_rows))


def preference_flip_accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_flip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("counterfactual_flip_id") and r.get("prediction") in {"A", "B"} and r.get("preferred") in {"A", "B"}:
            by_flip[str(r["counterfactual_flip_id"])].append(r)
    eligible = [v for v in by_flip.values() if len({x.get("preferred") for x in v}) >= 2]
    if not eligible:
        return _metric(None, 0)
    correct = sum(all(x["prediction"] == x["preferred"] for x in group) for group in eligible)
    return _metric(correct / len(eligible), len(eligible))


def flip_consistency_rate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_flip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("counterfactual_flip_id") and r.get("prediction") in {"A", "B"}:
            by_flip[str(r["counterfactual_flip_id"])].append(r)
    eligible = [v for v in by_flip.values() if len(v) >= 2]
    if not eligible:
        return _metric(None, 0)
    consistent = sum(len({x["prediction"] for x in group}) > 1 for group in eligible)
    return _metric(consistent / len(eligible), len(eligible))


def language_sensitivity_score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("pair_id") and r.get("prediction_score") is not None:
            by_pair[str(r["pair_id"])].append(r)
    diffs = []
    for group in by_pair.values():
        scores = [float(x["prediction_score"]) for x in group]
        if len(scores) >= 2:
            diffs.append(max(scores) - min(scores))
    if not diffs:
        return _metric(None, 0)
    return _metric(sum(diffs) / len(diffs), len(diffs))


def aggregate_metrics(pred_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {
        "standard_accuracy": standard_accuracy(pred_rows),
        "PFA": preference_flip_accuracy(pred_rows),
        "FCR": flip_consistency_rate(pred_rows),
        "LSS": language_sensitivity_score(pred_rows),
        "per_axis": {},
    }
    axes = sorted({r.get("axis") for r in pred_rows if r.get("axis")})
    for axis in axes:
        subset = [r for r in pred_rows if r.get("axis") == axis]
        out["per_axis"][axis] = {
            "standard_accuracy": standard_accuracy(subset),
            "PFA": preference_flip_accuracy(subset),
        }
    return out

