from __future__ import annotations

"""Build paper/tables/table_vlm.tex from GPT-4o (and Qwen, if available)
prediction files."""

import argparse
import collections
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cf_pref_learning.utils.io import read_jsonl


SPLITS = ["test_seen", "test_heldout_lexical", "test_heldout_camera",
          "test_heldout_color", "test_heldout_spatial", "test_hard_negatives"]
SHORT = {"test_seen": "Seen", "test_heldout_lexical": "Lex",
         "test_heldout_camera": "Cam", "test_heldout_color": "Color",
         "test_heldout_spatial": "Spatial", "test_hard_negatives": "HardNeg"}


def _pfa(rows: list[dict[str, Any]], strict: bool = False) -> tuple[float, int]:
    by_flip: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        if (r.get("counterfactual_flip_id") and r.get("preferred") in {"A", "B"}):
            if strict:
                by_flip[str(r["counterfactual_flip_id"])].append(r)
            else:
                if r.get("prediction") in {"A", "B"}:
                    by_flip[str(r["counterfactual_flip_id"])].append(r)
    eligible = [g for g in by_flip.values() if len({x.get("preferred") for x in g}) >= 2]
    if not eligible:
        return float("nan"), 0
    correct = sum(1 for g in eligible if all(x.get("prediction") == x.get("preferred") for x in g))
    return correct / len(eligible), len(eligible)


def _std_acc(rows: list[dict[str, Any]]) -> tuple[float, int]:
    eval_rows = [r for r in rows if r.get("preferred") in {"A", "B"} and r.get("prediction") in {"A", "B"}]
    if not eval_rows:
        return float("nan"), 0
    return sum(r["prediction"] == r["preferred"] for r in eval_rows) / len(eval_rows), len(eval_rows)


def _centroid_color_axis(root: Path) -> dict[str, float]:
    """Mean across 3 seeds of color-axis PFA per split from the engineered centroid Base-Pref."""
    import statistics
    seeds = [42, 1337, 2024]
    out: dict[str, list[float]] = {}
    for seed in seeds:
        path = root / f"outputs/e2_main/summary_base_pref_no_axis_seed{seed}.json"
        if not path.exists():
            continue
        s = json.loads(path.read_text())
        for split in SPLITS:
            v = (s.get("metrics", {}).get("by_split", {}).get(split, {})
                 .get("per_axis", {}).get("color", {}).get("PFA", {}).get("value"))
            if v is not None:
                out.setdefault(split, []).append(v)
    return {k: statistics.mean(v) for k, v in out.items()}


def _swap_consistency(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return float("nan")
    cons = sum(1 for r in rows if r.get("swap_consistent"))
    return cons / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/path/to/project")
    parser.add_argument("--gpt4o-dir", default="outputs/phase1_gpt4o")
    parser.add_argument("--qwen-dir", default="outputs/phase1_qwen")
    parser.add_argument("--output", default="paper/tables/table_vlm.tex")
    args = parser.parse_args()
    root = Path(args.project_root)
    gpt4o_dir = root / args.gpt4o_dir
    qwen_dir = root / args.qwen_dir

    judges = []
    if gpt4o_dir.exists():
        judges.append(("GPT-4o", gpt4o_dir, "gpt-4o-2024-11-20"))
    if qwen_dir.exists():
        judges.append(("Qwen2.5-VL-7B", qwen_dir, "Qwen2.5-VL-7B-Instruct"))

    table_rows = []
    for name, d, _ in judges:
        all_rows = []
        for s in SPLITS:
            path = d / f"predictions_{s}.jsonl"
            if path.exists():
                rows = read_jsonl(path)
                for r in rows:
                    r2 = dict(r); r2.setdefault("split", s); all_rows.append(r2)
        if not all_rows:
            continue
        # Per split color-axis PFA + overall std/PFA
        per_split = {}
        for s in SPLITS:
            sub = [r for r in all_rows if r.get("split") == s]
            color = [r for r in sub if r.get("axis") == "color"]
            std_v, _ = _std_acc(sub)
            pfa_v, _ = _pfa(sub)
            color_pfa_v, color_n = _pfa(color)
            swap_v = _swap_consistency(sub)
            per_split[s] = {"std": std_v, "pfa": pfa_v, "color_pfa": color_pfa_v, "color_n": color_n, "swap": swap_v}
        table_rows.append({"name": name, "per_split": per_split})

    # Compose LaTeX table - keep it compact
    out_lines = [
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{External vision-language judges on CF-PrefBench v3, restricted to the \texttt{axis=color} subset of each split (n=14-72 evaluable flips per split, sums per row). PFA uses the standard preference-flip-accuracy metric (rows with \texttt{Tie} predictions are excluded; the smaller denominator is reported). Swap consistency is the fraction of rows on which the original-order and swapped-order answers agreed.}",
        r"\label{tab:vlm}",
        r"\begin{tabular}{l" + "c" * len(SPLITS) + r"cc}",
        r"\toprule",
        r"Judge & " + " & ".join(SHORT[s] for s in SPLITS) + r" & overall (color-axis avg) & swap consistency \\",
        r"\midrule",
    ]
    for row in table_rows:
        cells = [row["name"]]
        cas = []
        swaps = []
        for s in SPLITS:
            ca = row["per_split"][s].get("color_pfa")
            n = row["per_split"][s].get("color_n")
            sw = row["per_split"][s].get("swap")
            if ca == ca and ca is not None:
                cells.append(f"${ca:.3f}^{{n={n}}}$")
                cas.append(ca)
            else:
                cells.append("--")
            if sw == sw and sw is not None:
                swaps.append(sw)
        avg_ca = sum(cas) / len(cas) if cas else float("nan")
        avg_sw = sum(swaps) / len(swaps) if swaps else float("nan")
        cells.append(f"${avg_ca:.3f}$" if avg_ca == avg_ca else "--")
        cells.append(f"${avg_sw:.3f}$" if avg_sw == avg_sw else "--")
        out_lines.append(" & ".join(cells) + r" \\")
    # Add the centroid baseline reference row from outputs/e2_main/summary files
    centroid_per_split = _centroid_color_axis(root)
    cells = ["Base-Pref centroid (reference)"]
    cas_ref = []
    for s in SPLITS:
        v = centroid_per_split.get(s)
        if v is not None:
            cells.append(f"${v:.3f}$")
            cas_ref.append(v)
        else:
            cells.append("--")
    avg_ref = sum(cas_ref) / len(cas_ref) if cas_ref else float("nan")
    cells.append(f"$\\mathbf{{{avg_ref:.3f}}}$" if avg_ref == avg_ref else "--")
    cells.append("--")
    out_lines += [r"\midrule",
                  " & ".join(cells) + r" \\",
                  r"\bottomrule",
                  r"\end{tabular}", r"\end{table*}"]
    (root / args.output).write_text("\n".join(out_lines))
    print(f"wrote {args.output}")
    # Also dump a JSON for easy programmatic access
    (root / args.output.replace(".tex", "_data.json")).write_text(json.dumps(table_rows, indent=2))


if __name__ == "__main__":
    main()
