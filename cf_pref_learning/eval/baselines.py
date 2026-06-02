from __future__ import annotations

from typing import Any

from .eval_impossible import impossible_premise_metrics
from .metrics import aggregate_metrics


def summarize_baseline_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate baseline predictions after scoring.

    This helper is intentionally post-hoc: fields such as axis, pair_id, and
    counterfactual ids may be used here for reporting metrics, but they must not
    be used by model feature builders.
    """

    metrics = aggregate_metrics(rows)
    metrics["impossible_premise"] = impossible_premise_metrics(rows)
    return metrics
