from __future__ import annotations

"""Generate the unified Table 1 (paper/tables/table_main.tex) and the
diagnostic Table (paper/tables/table_diag.tex) from per-seed summaries.

Sources:
  - outputs/e2_main/summary_*.json
  - outputs/track_a_frame_clip/summary_*.json
  - outputs/track_a_combined/summary_*.json
  - outputs/track_a_centroid_sanity/summary_*.json
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


SPLITS = ["test_seen", "test_heldout_lexical", "test_heldout_camera",
          "test_heldout_color", "test_heldout_spatial", "test_hard_negatives"]
SHORT = {"test_seen": "Seen", "test_heldout_lexical": "Lex",
         "test_heldout_camera": "Cam", "test_heldout_color": "Color",
         "test_heldout_spatial": "Spatial", "test_hard_negatives": "HardNeg"}


METHODS_MAIN = [
    ("Base-Pref", "outputs/e2_main/summary_base_pref_no_axis_seed{seed}.json", [42, 1337, 2024]),
    ("CPL-Pref (CE)", "outputs/e2_main/summary_cpl_pref_ce_seed{seed}.json", [42, 1337, 2024]),
    ("CPL-Pref (CE+M)", "outputs/e2_main/summary_cpl_pref_ce_margin_seed{seed}.json", [42, 1337, 2024]),
    ("CPL-Pref (CE+M+Para)", "outputs/e2_main/summary_cpl_pref_ce_margin_para_seed{seed}.json", [42, 1337, 2024]),
    ("CPL-Pref (Para-only)", "outputs/e2_main/summary_cpl_pref_para_only_seed{seed}.json", [42, 1337, 2024]),
    (r"CPL-Pref (GroupContrast, \emph{ours})", "outputs/e3_groupcontrast/summary_cpl_pref_groupcontrast_seed{seed}.json", [42, 1337, 2024]),
    ("Track A: frame-CLIP", "outputs/track_a_frame_clip/summary_base_pref_no_axis_clipframe_seed{seed}.json", [42, 1337, 2024]),
    ("Track A: centroid+CLIP", "outputs/track_a_combined/summary_base_pref_no_axis_combined_seed{seed}.json", [42, 1337, 2024]),
    (r"Track A: cross-attn CLIP (\emph{ours})", "outputs/track_a_crossattn/summary_base_pref_crossattn_clip_seed{seed}.json", [42, 1337, 2024]),
    ("Track A: centroid (sanity)", "outputs/track_a_centroid_sanity/summary_base_pref_no_axis_centroid_seed{seed}.json", [42, 1337, 2024]),
]

METHODS_DIAG = [
    (r"CPL-Pref (CE, $\lambda_{\rm cf}{=}0.1$)", "outputs/e2_main/summary_cpl_pref_ce_smalllambda_seed{seed}.json", [42]),
    (r"CPL-Pref (CE, $\lambda_{\rm cf}{=}0.05$)", "outputs/e2_main/summary_cpl_pref_ce_tinylambda_seed{seed}.json", [42]),
]


def _load_summaries(root: Path, methods: list[tuple[str, str, list[int]]]) -> list[dict[str, Any]]:
    out = []
    for name, glob, seeds in methods:
        summaries = []
        for seed in seeds:
            path = root / glob.format(seed=seed)
            if path.exists():
                summaries.append(json.loads(path.read_text()))
        if summaries:
            out.append({"name": name, "seeds": seeds, "summaries": summaries})
    return out


def _stat(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mn = sum(values) / len(values)
    sd = statistics.pstdev(values) if len(values) >= 2 else 0.0
    return mn, sd


def _fmt(m: float, s: float, fmt: str = ".3f") -> str:
    if m != m:  # nan
        return "--"
    return f"${m:{fmt}} \\pm {s:{fmt}}$"


def build_main_table(root: Path, methods: list[tuple[str, str, list[int]]]) -> str:
    rows = _load_summaries(root, methods)
    # Per-split PFA only (no Std) so table fits on page
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{CF-PrefBench v3 main results: \textbf{preference-flip accuracy (PFA)} per split,"
        r" 7 head-side CPL variants (top) and 3 feature-side Track A variants (bottom)."
        r" All variants share the same 3-layer MLP head and same 3 random seeds \{42, 1337, 2024\}."
        r" The rightmost column is the per-axis PFA of the \texttt{axis=color} subset within"
        r" \texttt{test\_heldout\_color} (the held-out compositional color binding test) — this is"
        r" the metric the CPL claim is about.}",
        r"\label{tab:e2_main_v3}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "c" * len(SPLITS) + r"c}",
        r"\toprule",
        r"Variant & " + " & ".join(SHORT[s] for s in SPLITS) + r" & \textbf{Color-axis} \\",
        r"\midrule",
    ]
    for row in rows:
        cells = [row["name"]]
        for s in SPLITS:
            pfa_vals = []
            for sm in row["summaries"]:
                m = sm.get("metrics", {}).get("by_split", {}).get(s, {})
                v_pfa = m.get("PFA", {}).get("value")
                if v_pfa is not None: pfa_vals.append(v_pfa)
            mp, sp = _stat(pfa_vals)
            cells.append(_fmt(mp, sp, ".2f"))
        # color-axis PFA on test_heldout_color
        ca_vals = []
        for sm in row["summaries"]:
            v = (sm.get("metrics", {}).get("by_split", {}).get("test_heldout_color", {})
                 .get("per_axis", {}).get("color", {}).get("PFA", {}).get("value"))
            if v is not None: ca_vals.append(v)
        mc, sc = _stat(ca_vals)
        cells.append(r"\textbf{" + _fmt(mc, sc, ".2f") + r"}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def build_diag_table(root: Path) -> str:
    rows = _load_summaries(root, METHODS_DIAG)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Diagnostic CPL variants on seed 42, designed to test whether shrinking $\lambda_{\rm cf}$ unlocks an improvement.}",
        r"\label{tab:e2_diag}",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Variant & val PFA & heldout-avg PFA & Color-axis PFA \\",
        r"\midrule",
    ]
    for row in rows:
        sm = row["summaries"][0]
        v_pfa = sm.get("metrics", {}).get("by_split", {}).get("val", {}).get("PFA", {}).get("value")
        heldout_pfa = []
        for s in ["test_heldout_lexical", "test_heldout_camera", "test_heldout_color",
                   "test_heldout_spatial", "test_hard_negatives"]:
            v = sm.get("metrics", {}).get("by_split", {}).get(s, {}).get("PFA", {}).get("value")
            if v is not None: heldout_pfa.append(v)
        havg = sum(heldout_pfa) / len(heldout_pfa) if heldout_pfa else float("nan")
        ca = (sm.get("metrics", {}).get("by_split", {}).get("test_heldout_color", {})
              .get("per_axis", {}).get("color", {}).get("PFA", {}).get("value"))
        lines.append(f"{row['name']} & ${v_pfa:.3f}$ & ${havg:.3f}$ & ${ca:.3f}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/path/to/project")
    args = parser.parse_args()
    root = Path(args.project_root)

    main_tex = build_main_table(root, METHODS_MAIN)
    (root / "paper/tables/table_main.tex").write_text(main_tex)
    diag_tex = build_diag_table(root)
    (root / "paper/tables/table_diag.tex").write_text(diag_tex)
    print(f"wrote paper/tables/table_main.tex, paper/tables/table_diag.tex")


if __name__ == "__main__":
    main()
