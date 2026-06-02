"""Analyze VLM judge results: per-binding-token accuracy.

The binding token differs by axis:
  motion → first word (verb): transit, shift, convey, scoot
  size   → size adjective: miniature, petite, colossal, gigantic
  speed  → speed adverb: briskly, speedily, sluggishly, gradually
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cf_pref_learning.utils.io import read_jsonl, write_json


# Per-split, what binding-token are we looking for?
# Mapping: split -> (axis_class, expected_binding_tokens)
SPLIT_INFO = {
    "test_heldout_lexical":          {"axis": "motion", "tokens": ["shift", "convey", "transit"]},
    "test_heldout_lexical_scoot":    {"axis": "motion", "tokens": ["scoot"]},
    "test_heldout_size_miniature":   {"axis": "size_SMALL", "tokens": ["miniature"]},
    "test_heldout_size_petite":      {"axis": "size_SMALL", "tokens": ["petite"]},
    "test_heldout_size_colossal":    {"axis": "size_BIG",   "tokens": ["colossal"]},
    "test_heldout_size_gigantic":    {"axis": "size_BIG",   "tokens": ["gigantic"]},
    "test_heldout_speed_briskly":    {"axis": "speed_FAST", "tokens": ["briskly"]},
    "test_heldout_speed_sluggishly": {"axis": "speed_SLOW", "tokens": ["sluggishly"]},
    "test_heldout_speed_speedily":   {"axis": "speed_FAST", "tokens": ["speedily"]},
    "test_heldout_speed_gradually":  {"axis": "speed_SLOW", "tokens": ["gradually"]},
}

# LIB v0 reference numbers from Exp 3, 3b, 4 (3-seed means)
LIB_REF = {
    "transit":     {"b32": 0.500, "l14": 0.917, "cos_b32": 0.915, "class": "motion below-cliff"},
    "scoot":       {"b32": 0.679, "l14": 0.905, "cos_b32": 0.927, "class": "motion below-cliff"},
    "convey":      {"b32": 0.929, "l14": 0.929, "cos_b32": 0.938, "class": "motion above-cliff"},
    "shift":       {"b32": 0.917, "l14": 0.905, "cos_b32": 0.968, "class": "motion above-cliff"},
    "miniature":   {"b32": 0.397, "l14": 0.817, "cos_b32": 0.918, "class": "size SMALL"},
    "petite":      {"b32": 0.492, "l14": 0.770, "cos_b32": 0.917, "class": "size SMALL"},
    "colossal":    {"b32": 0.921, "l14": 0.810, "cos_b32": 0.898, "class": "size BIG"},
    "gigantic":    {"b32": 0.929, "l14": 0.810, "cos_b32": 0.918, "class": "size BIG"},
    "briskly":     {"b32": 0.048, "l14": 0.317, "cos_b32": 0.937, "class": "speed FAST"},
    "speedily":    {"b32": 0.183, "l14": 0.460, "cos_b32": 0.942, "class": "speed FAST"},
    "sluggishly":  {"b32": 0.960, "l14": 0.992, "cos_b32": 0.932, "class": "speed SLOW"},
    "gradually":   {"b32": 0.698, "l14": 1.000, "cos_b32": 0.938, "class": "speed SLOW"},
}


def extract_binding_token(instr: str, split: str) -> str | None:
    info = SPLIT_INFO.get(split)
    if not info: return None
    for tok in info["tokens"]:
        if tok in instr:
            return tok
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm-name", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path("/path/to/project")

    by_token = defaultdict(list)  # token -> [(preferred, prediction)]
    for split in SPLIT_INFO:
        pred_path = root / args.input_dir / f"predictions_{split}.jsonl"
        if not pred_path.exists():
            continue
        preds = read_jsonl(pred_path)
        preds = [p for p in preds if p.get("prediction") in {"A", "B"} and p.get("preferred") in {"A", "B"}]
        if split == "test_heldout_lexical":
            preds = [p for p in preds if p.get("axis") == "motion_sequence"]
        for p in preds:
            tok = extract_binding_token(p["instruction"], split)
            if tok:
                by_token[tok].append((p["preferred"], p["prediction"]))

    # Aggregate
    rows = []
    for tok, items in sorted(by_token.items()):
        nc = sum(1 for pref, pred in items if pref == pred)
        vlm_acc = nc / len(items)
        ref = LIB_REF.get(tok, {})
        rows.append({
            "token": tok, "n": len(items), "vlm_acc": vlm_acc,
            "lib_b32": ref.get("b32"), "lib_l14": ref.get("l14"),
            "cos_b32": ref.get("cos_b32"), "class": ref.get("class", "?"),
        })

    # Order rows by class
    class_order = ["motion above-cliff", "motion below-cliff",
                    "size BIG", "size SMALL", "speed FAST", "speed SLOW"]
    rows.sort(key=lambda r: (class_order.index(r["class"]) if r["class"] in class_order else 99, r["token"]))

    # Print
    print(f"\n=== {args.vlm_name} per-binding-token cliff analysis ===\n")
    print(f"{'token':12s} {'n':>4s} {'VLM':>7s} {'B/32':>7s} {'L/14':>7s} {'cos':>7s} {'class':>22s}")
    for r in rows:
        b32 = f"{r['lib_b32']:.3f}" if r['lib_b32'] is not None else "---"
        l14 = f"{r['lib_l14']:.3f}" if r['lib_l14'] is not None else "---"
        cos = f"{r['cos_b32']:.4f}" if r['cos_b32'] is not None else "---"
        print(f"{r['token']:12s} {r['n']:>4d} {r['vlm_acc']:>7.3f} {b32:>7s} {l14:>7s} {cos:>7s} {r['class']:>22s}")

    # Cliff-specific summary
    cliff_tokens = ["transit", "scoot", "miniature", "petite", "briskly", "speedily"]
    print(f"\n=== Cliff tokens: VLM vs LIB B/32 ===\n")
    for v in cliff_tokens:
        r = next((rr for rr in rows if rr["token"] == v), None)
        if r and r["lib_b32"] is not None:
            delta = r["vlm_acc"] - r["lib_b32"]
            print(f"  {v:12s}  VLM={r['vlm_acc']:.3f}  LIB_B32={r['lib_b32']:.3f}  Δ={delta:+.3f}")

    # Write markdown
    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {args.vlm_name} Per-Binding-Token Cliff Analysis",
        "",
        "Each row is the VLM's row-level accuracy on a single held-out binding token,",
        "compared with LIB v0 (B/32 and L/14) reference numbers from Exp 3-4.",
        "",
        "| Token | n | VLM | LIB B/32 | LIB L/14 | cos(B/32) | Class |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        b32 = f"{r['lib_b32']:.3f}" if r['lib_b32'] is not None else "—"
        l14 = f"{r['lib_l14']:.3f}" if r['lib_l14'] is not None else "—"
        cos = f"{r['cos_b32']:.4f}" if r['cos_b32'] is not None else "—"
        lines.append(f"| {r['token']} | {r['n']} | {r['vlm_acc']:.3f} | {b32} | {l14} | {cos} | {r['class']} |")
    lines.append("")
    lines.append("## Cliff tokens (VLM − LIB B/32)")
    lines.append("")
    lines.append("| Token | VLM | LIB B/32 | Δ |")
    lines.append("| --- | ---: | ---: | ---: |")
    for v in cliff_tokens:
        r = next((rr for rr in rows if rr["token"] == v), None)
        if r and r["lib_b32"] is not None:
            d = r["vlm_acc"] - r["lib_b32"]
            lines.append(f"| {v} | {r['vlm_acc']:.3f} | {r['lib_b32']:.3f} | {d:+.3f} |")
    out_path.write_text("\n".join(lines))
    print(f"\nwrote {out_path}")
    write_json(out_path.with_suffix(".json"), {"vlm": args.vlm_name, "rows": rows})


if __name__ == "__main__":
    main()
