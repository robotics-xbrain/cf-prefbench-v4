from __future__ import annotations

from collections import defaultdict
from typing import Any


def _status(violations: list[Any]) -> str:
    return "fail" if violations else "pass"


def audit_leakage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "verdict": "blocked",
            "reason": "No CF-PrefBench examples are available.",
            "checks": {},
        }
    train = [r for r in rows if r.get("split") == "train"]
    test_seen = [r for r in rows if r.get("split") == "test_seen"]
    heldout = [r for r in rows if r.get("split") == "test_heldout_lexical"]
    test_all = [r for r in rows if str(r.get("split", "")).startswith("test_")]

    train_lex = {x for r in train for x in r.get("lexical_items", [])}
    heldout_lex = {x for r in heldout for x in r.get("lexical_items", [])}
    train_pairs = {r.get("pair_id") for r in train}
    test_pairs = {r.get("pair_id") for r in test_all}
    train_inst_pair = {(r.get("instruction"), r.get("pair_id")) for r in train}
    test_inst_pair = {(r.get("instruction"), r.get("pair_id")) for r in test_all}
    train_flip = {r.get("counterfactual_flip_id") for r in train}
    heldout_flip = {r.get("counterfactual_flip_id") for r in heldout}

    cf_by_split: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        cf_by_split[str(r.get("split"))].add(str(r.get("counterfactual_flip_id")))

    checks = {
        "heldout_lexical_flip_in_train": {
            "status": _status(sorted(train_lex & heldout_lex)),
            "violations": sorted(train_lex & heldout_lex),
        },
        "heldout_video_pair_in_train": {
            "status": _status(sorted(train_pairs & test_pairs)),
            "violations": sorted(x for x in (train_pairs & test_pairs) if x is not None),
        },
        "exact_instruction_pair_leakage": {
            "status": _status(sorted(train_inst_pair & test_inst_pair)),
            "violations": [{"instruction": a, "pair_id": b} for a, b in sorted(train_inst_pair & test_inst_pair)],
        },
        "pair_id_leakage_across_train_test": {
            "status": _status(sorted(train_pairs & test_pairs)),
            "violations": sorted(x for x in (train_pairs & test_pairs) if x is not None),
        },
        "counterfactual_flip_id_leakage_train_test_heldout_lexical": {
            "status": _status(sorted(train_flip & heldout_flip)),
            "violations": sorted(x for x in (train_flip & heldout_flip) if x is not None),
        },
    }
    verdict = "pass" if all(check["status"] == "pass" for check in checks.values()) else "fail"
    split_names = sorted({str(r.get("split")) for r in rows})
    return {"verdict": verdict, "checks": checks, "split_counts": {s: len([r for r in rows if r.get("split") == s]) for s in split_names}}
