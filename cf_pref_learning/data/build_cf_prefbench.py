from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .converters import convert_sources_to_cf_prefbench
from .discover_sources import discover_project_sources, summarize_external_paths
from .export_human_check import export_human_check
from .leakage import audit_leakage
from .schemas import issue_to_dict, validate_example
from .split import split_existing_or_assign
from .split import SPLITS
from ..eval.metrics import aggregate_metrics
from ..utils.io import ensure_dir, write_json, write_jsonl

DEFAULT_EXTERNAL_HINTS = [
    "/path/to/doctor_program/chapter_v1_core/datasets",
    "/path/to/doctor_program/chapter_v1/datasets",
    "/path/to/papers/nips_0/counterfactual_curation/data",
    "/path/to/bop_datasets",
]


def _stats(rows: list[dict[str, Any]], validation_issues: list[dict[str, Any]]) -> dict[str, Any]:
    split_counts = Counter(r.get("split") for r in rows)
    axis_counts = Counter(r.get("axis") for r in rows)
    label_counts = Counter(r.get("preferred") for r in rows)
    per_split_axis: dict[str, Counter] = defaultdict(Counter)
    per_axis_label: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        per_split_axis[str(r.get("split"))][str(r.get("axis"))] += 1
        per_axis_label[str(r.get("axis"))][str(r.get("preferred"))] += 1
    return {
        "n_examples": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "axis_counts": dict(sorted(axis_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "per_split_axis": {k: dict(v) for k, v in sorted(per_split_axis.items())},
        "per_axis_label": {k: dict(v) for k, v in sorted(per_axis_label.items())},
        "n_counterfactual_groups": len({r.get("counterfactual_group_id") for r in rows}),
        "n_counterfactual_flips": len({r.get("counterfactual_flip_id") for r in rows}),
        "n_pair_ids": len({r.get("pair_id") for r in rows}),
        "lexical_item_counts": dict(sorted(Counter(x for r in rows for x in r.get("lexical_items", [])).items())),
        "validation_issue_count": len(validation_issues),
        "validation_issues_sample": validation_issues[:100],
    }


def run_e0(project_root: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    root = Path(project_root)
    out_dir = root / "outputs" / "e0_data_audit"
    ensure_dir(out_dir)
    external_hints = config.get("external_audit_paths", DEFAULT_EXTERNAL_HINTS)
    project_sources = discover_project_sources(root)
    external_sources = summarize_external_paths(external_hints)
    rows, converter_notes = convert_sources_to_cf_prefbench(root, [project_sources] + external_sources)
    rows = split_existing_or_assign(rows, int(config.get("seed", 42)))

    validation_issues = []
    for row in rows:
        validation_issues.extend(issue_to_dict(x) for x in validate_example(row, root, check_video=bool(config.get("check_video_paths", True))))
    valid = rows and not validation_issues
    for split in SPLITS:
        split_rows = [r for r in rows if r.get("split") == split]
        write_jsonl(root / "data" / "cf_prefbench" / f"{split}.jsonl", split_rows)
    stats = _stats(rows, validation_issues)
    stats.update({
        "project_sources": project_sources,
        "external_sources": external_sources,
        "converter_notes": converter_notes,
        "real_data_available": bool(rows),
        "e0_status": "pass" if valid else "blocked",
    })
    leakage = audit_leakage(rows)
    human = export_human_check(rows, out_dir / "human_check_samples.jsonl", seed=int(config.get("seed", 42))) if rows else {"exported": 0, "limitation": "No examples available."}
    stats["human_check_export"] = human
    write_json(out_dir / "toy_validation.json", _toy_validation(root))
    write_json(out_dir / "data_statistics.json", stats)
    write_json(out_dir / "leakage_report.json", leakage)
    _write_markdown_reports(root, stats, leakage)
    return {"statistics": stats, "leakage": leakage}


def _write_markdown_reports(root: Path, stats: dict[str, Any], leakage: dict[str, Any]) -> None:
    out_dir = root / "outputs" / "e0_data_audit"
    data_audit = root / "DATA_SOURCE_AUDIT.md"
    lines = [
        "# Data Source Audit",
        "",
        "Date: 2026-05-15",
        f"Project root: `{root}`",
        "",
        "## In-Project Sources",
        "",
        _source_block(stats["project_sources"]),
        "",
        "## External Sources Observed",
        "",
    ]
    for source in stats["external_sources"]:
        lines.extend([_source_block(source), ""])
    lines.extend([
        "## CF-PrefBench Readiness",
        "",
        f"Real in-root CF-PrefBench examples available: `{stats['real_data_available']}`",
        f"Validation issue count: `{stats['validation_issue_count']}`",
        "",
        "External robotics datasets were not converted because the current task requires all code, logs, outputs, and benchmark artifacts under the project root, and the observed external sources do not expose the full required tuple of video pairs, language instructions, counterfactual preference labels, and task metadata.",
    ])
    data_audit.write_text("\n".join(lines) + "\n", encoding="utf-8")

    e0_summary = [
        "# E0 Summary",
        "",
        f"Status: `{stats['e0_status']}`",
        f"Examples: `{stats['n_examples']}`",
        f"Leakage verdict: `{leakage.get('verdict')}`",
        f"Human-check samples exported: `{stats['human_check_export']['exported']}`",
        "",
        "E1 training must not start unless real CF-PrefBench examples are present and the leakage verdict is `pass`.",
    ]
    (out_dir / "E0_SUMMARY.md").write_text("\n".join(e0_summary) + "\n", encoding="utf-8")

    review = [
        "# E0 Self Audit",
        "",
        "- Schema validation implemented and run.",
        "- Video path existence check implemented.",
        "- Axis and label balance statistics implemented.",
        "- Leakage audit includes held-out lexical items, video pair leakage, exact instruction-pair leakage, pair_id leakage, and counterfactual_flip_id leakage.",
        "- No final numbers were manually typed into tables.",
        f"- Hard stop triggered: `{stats['e0_status'] != 'pass' or leakage.get('verdict') != 'pass'}`.",
    ]
    (root / "outputs" / "reviews" / "E0_SELF_AUDIT.md").write_text("\n".join(review) + "\n", encoding="utf-8")


def _toy_validation(root: Path) -> dict[str, Any]:
    rows = [
        {
            "example_id": "toy_1",
            "pair_id": "toy_pair_1",
            "video_a": "toy/a.mp4",
            "video_b": "toy/b.mp4",
            "instruction": "push the red block",
            "preferred": "A",
            "axis": "color",
            "counterfactual_group_id": "toy_group_1",
            "counterfactual_flip_id": "toy_flip_1",
            "lexical_items": ["red"],
            "split": "test_heldout_lexical",
            "paraphrase_group_id": None,
            "metadata": {},
            "prediction": "A",
            "prediction_score": 1.0,
        },
        {
            "example_id": "toy_2",
            "pair_id": "toy_pair_1_cf",
            "video_a": "toy/a.mp4",
            "video_b": "toy/b.mp4",
            "instruction": "push the blue block",
            "preferred": "B",
            "axis": "color",
            "counterfactual_group_id": "toy_group_1",
            "counterfactual_flip_id": "toy_flip_1",
            "lexical_items": ["blue"],
            "split": "test_heldout_lexical",
            "paraphrase_group_id": None,
            "metadata": {},
            "prediction": "B",
            "prediction_score": -1.0,
        },
    ]
    schema_issues = []
    for row in rows:
        schema_issues.extend(issue_to_dict(x) for x in validate_example(row, root, check_video=False))
    return {
        "purpose": "Toy-only validation of schema, leakage, and metrics code. Not paper evidence.",
        "schema_issue_count": len(schema_issues),
        "leakage": audit_leakage(rows),
        "metrics": aggregate_metrics(rows),
    }


def _source_block(source: dict[str, Any]) -> str:
    missing = source.get("missing_for_cf_prefbench", [])
    return "\n".join([
        f"### `{source.get('root')}`",
        "",
        f"- Exists: `{source.get('exists')}`",
        f"- File count: `{source.get('file_count')}`",
        f"- Contains trajectory/video pairs: `{source.get('contains_video_files')}`",
        f"- Contains language instructions: `{source.get('contains_language_instructions_detected')}`",
        f"- Contains preference labels: `{source.get('contains_preference_labels_detected')}`",
        f"- Contains task metadata: `{source.get('contains_task_metadata_detected')}`",
        f"- Can support CF-PrefBench construction directly: `{source.get('can_support_cf_prefbench_construction')}`",
        f"- Missing if not: `{', '.join(missing) if missing else 'none detected by filename audit'}`",
        f"- Counts by extension: `{source.get('counts_by_extension')}`",
    ])
