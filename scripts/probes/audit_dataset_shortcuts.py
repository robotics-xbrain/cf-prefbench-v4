from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cf_pref_learning.utils.io import ensure_dir, read_jsonl, write_json


SPLITS = [
    "train",
    "val",
    "test_seen",
    "test_heldout_lexical",
    "test_heldout_camera",
    "test_heldout_color",
    "test_heldout_spatial",
    "test_hard_negatives",
]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _load_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        rows.extend(read_jsonl(root / "data" / "cf_prefbench" / f"{split}.jsonl"))
    return rows


def _dist(rows: list[dict[str, Any]], key_fn: Any) -> dict[str, dict[str, int]]:
    out: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        keys = key_fn(row)
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            out[str(key)][str(row.get("preferred"))] += 1
    return {k: dict(v) for k, v in sorted(out.items())}


def _majority(counter: Counter[str]) -> str:
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _text_only_accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    train = [r for r in rows if r.get("split") == "train" and r.get("preferred") in {"A", "B"}]
    eval_rows = [r for r in rows if r.get("split") != "train" and r.get("preferred") in {"A", "B"}]
    by_instruction: dict[str, Counter[str]] = defaultdict(Counter)
    by_token: dict[str, Counter[str]] = defaultdict(Counter)
    by_axis: dict[str, Counter[str]] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()
    for row in train:
        label = str(row["preferred"])
        by_instruction[row["instruction"]][label] += 1
        by_axis[row["axis"]][label] += 1
        global_counts[label] += 1
        for tok in _tokens(row["instruction"]):
            by_token[tok][label] += 1
    if not eval_rows or not global_counts:
        return {"available": False, "n": 0, "value": None, "method": "instruction/token/axis majority from train"}
    correct = 0
    sources = Counter()
    global_pred = _majority(global_counts)
    for row in eval_rows:
        source = "global"
        counter = Counter()
        if row["instruction"] in by_instruction:
            counter = by_instruction[row["instruction"]]
            source = "instruction"
        else:
            for tok in _tokens(row["instruction"]):
                counter.update(by_token.get(tok, Counter()))
            if counter:
                source = "token"
            elif row["axis"] in by_axis:
                counter = by_axis[row["axis"]]
                source = "axis"
        pred = _majority(counter) if counter else global_pred
        correct += pred == row["preferred"]
        sources[source] += 1
    return {
        "available": True,
        "n": len(eval_rows),
        "value": correct / len(eval_rows),
        "method": "instruction/token/axis majority from train",
        "prediction_sources": dict(sources),
    }


def _balance_by_id(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        groups[str(row.get(field))][str(row.get("preferred"))] += 1
    sizes = [sum(c.values()) for c in groups.values()]
    ab_balanced = {
        k: c.get("A", 0) == c.get("B", 0)
        for k, c in groups.items()
        if c.get("A", 0) or c.get("B", 0)
    }
    return {
        "n_groups": len(groups),
        "min_size": min(sizes) if sizes else 0,
        "max_size": max(sizes) if sizes else 0,
        "mean_size": sum(sizes) / len(sizes) if sizes else 0,
        "n_ab_groups": len(ab_balanced),
        "n_ab_balanced": sum(ab_balanced.values()),
        "fraction_ab_balanced": sum(ab_balanced.values()) / len(ab_balanced) if ab_balanced else None,
        "sample_unbalanced": [k for k, ok in sorted(ab_balanced.items()) if not ok][:20],
    }


def _original_counterfactual_balance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        role = row.get("metadata", {}).get("cf_role", "unspecified")
        counts[str(role)][str(row.get("preferred"))] += 1
    return {k: dict(v) for k, v in sorted(counts.items())}


def audit(root: Path) -> dict[str, Any]:
    rows = _load_rows(root)
    impossible = [r for r in rows if r.get("axis") == "impossible_premise"]
    train_impossible = [r for r in impossible if r.get("split") == "train"]
    non_tie_impossible = [r for r in impossible if r.get("preferred") != "Tie"]
    included_impossible = [r for r in impossible if not r.get("metadata", {}).get("exclude_from_cpl_training")]
    return {
        "n_examples": len(rows),
        "split_counts": dict(Counter(str(r.get("split")) for r in rows)),
        "label_distribution_by_instruction": _dist(rows, lambda r: r.get("instruction", "")),
        "label_distribution_by_axis": _dist(rows, lambda r: r.get("axis", "")),
        "label_distribution_by_lexical_item": _dist(rows, lambda r: r.get("lexical_items", [])),
        "text_only_label_predictability": _text_only_accuracy(rows),
        "pair_id_balance": _balance_by_id(rows, "pair_id"),
        "counterfactual_group_id_balance": _balance_by_id(rows, "counterfactual_group_id"),
        "counterfactual_flip_id_balance": _balance_by_id(rows, "counterfactual_flip_id"),
        "original_counterfactual_label_balance": _original_counterfactual_balance(rows),
        "paraphrase_group_count": len({r.get("paraphrase_group_id") for r in rows if r.get("paraphrase_group_id")}),
        "impossible_premise": {
            "n": len(impossible),
            "all_tie": not non_tie_impossible,
            "n_non_tie": len(non_tie_impossible),
            "n_train": len(train_impossible),
            "all_excluded_from_training": not included_impossible,
            "n_not_excluded_from_training": len(included_impossible),
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    pred = report["text_only_label_predictability"]
    pair = report["pair_id_balance"]
    cf = report["counterfactual_group_id_balance"]
    imp = report["impossible_premise"]
    lines = [
        "# Dataset Shortcut Audit",
        "",
        f"Examples: `{report['n_examples']}`",
        f"Splits: `{report['split_counts']}`",
        "",
        "## Text-Only Predictability",
        "",
        f"Train-derived text-only A/B accuracy on non-train A/B examples: `{pred.get('value')}` over `{pred.get('n')}` examples.",
        f"Prediction sources: `{pred.get('prediction_sources', {})}`",
        "",
        "## Balance Checks",
        "",
        f"Pair groups: `{pair['n_groups']}`; size range `{pair['min_size']}`-`{pair['max_size']}`; A/B balanced fraction `{pair['fraction_ab_balanced']}`.",
        f"Counterfactual groups: `{cf['n_groups']}`; size range `{cf['min_size']}`-`{cf['max_size']}`; A/B balanced fraction `{cf['fraction_ab_balanced']}`.",
        f"Original/counterfactual label balance: `{report['original_counterfactual_label_balance']}`",
        "",
        "## Impossible Premise",
        "",
        f"Impossible examples: `{imp['n']}`; all Tie: `{imp['all_tie']}`; train examples: `{imp['n_train']}`; all excluded from training: `{imp['all_excluded_from_training']}`.",
        "",
        "## Distributions",
        "",
        "Full label distributions by instruction, axis, and lexical item are in `shortcut_audit.json`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/path/to/project")
    args = parser.parse_args()
    root = Path(args.project_root)
    out_dir = ensure_dir(root / "outputs" / "e1_diagnostic")
    report = audit(root)
    write_json(out_dir / "shortcut_audit.json", report)
    (out_dir / "shortcut_audit.md").write_text(_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
