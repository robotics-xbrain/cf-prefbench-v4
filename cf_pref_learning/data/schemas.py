from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

VALID_PREFERRED = {"A", "B", "Tie"}
VALID_AXES = {"action", "object", "color", "spatial", "impossible_premise"}
VALID_SPLITS = {
    "train",
    "val",
    "test_seen",
    "test_heldout_lexical",
    "test_heldout_camera",
    "test_heldout_color",
    "test_heldout_spatial",
    "test_hard_negatives",
}
REQUIRED_FIELDS = {
    "example_id": str,
    "pair_id": str,
    "video_a": str,
    "video_b": str,
    "instruction": str,
    "preferred": str,
    "axis": str,
    "counterfactual_group_id": str,
    "counterfactual_flip_id": str,
    "lexical_items": list,
    "split": str,
    "metadata": dict,
}


@dataclass(frozen=True)
class ValidationIssue:
    example_id: str | None
    field: str
    message: str


def validate_example(example: dict[str, Any], project_root: str | Path, check_video: bool = True) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    example_id = example.get("example_id") if isinstance(example.get("example_id"), str) else None
    for field, expected in REQUIRED_FIELDS.items():
        if field not in example:
            issues.append(ValidationIssue(example_id, field, "missing required field"))
        elif not isinstance(example[field], expected):
            issues.append(ValidationIssue(example_id, field, f"expected {expected.__name__}"))
    if "paraphrase_group_id" in example and example["paraphrase_group_id"] is not None and not isinstance(example["paraphrase_group_id"], str):
        issues.append(ValidationIssue(example_id, "paraphrase_group_id", "expected string or null"))
    if example.get("preferred") not in VALID_PREFERRED:
        issues.append(ValidationIssue(example_id, "preferred", "must be A, B, or Tie"))
    if example.get("axis") not in VALID_AXES:
        issues.append(ValidationIssue(example_id, "axis", "unknown axis"))
    if example.get("split") not in VALID_SPLITS:
        issues.append(ValidationIssue(example_id, "split", "unknown split"))
    if not all(isinstance(x, str) for x in example.get("lexical_items", [])):
        issues.append(ValidationIssue(example_id, "lexical_items", "all lexical items must be strings"))
    if check_video:
        root = Path(project_root)
        for field in ("video_a", "video_b"):
            value = example.get(field)
            if isinstance(value, str):
                path = Path(value)
                if not path.is_absolute():
                    path = root / path
                if not path.exists():
                    issues.append(ValidationIssue(example_id, field, f"missing video path: {value}"))
    return issues


def issue_to_dict(issue: ValidationIssue) -> dict[str, Any]:
    return {"example_id": issue.example_id, "field": issue.field, "message": issue.message}
