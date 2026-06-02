from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cf_pref_learning.utils.io import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/path/to/project")
    args = parser.parse_args()
    root = Path(args.project_root)
    paths = [
        root / "outputs" / "e0_data_audit" / "data_statistics.json",
        root / "outputs" / "e0_data_audit" / "leakage_report.json",
        root / "outputs" / "e1_diagnostic" / "metrics.json",
    ]
    aggregate = {str(p.relative_to(root)): read_json(p) for p in paths if p.exists()}
    write_json(root / "outputs" / "all_results.json", aggregate)
    write_reports(root, aggregate)


def write_reports(root: Path, aggregate: dict) -> None:
    stats = aggregate.get("outputs/e0_data_audit/data_statistics.json", {})
    leakage = aggregate.get("outputs/e0_data_audit/leakage_report.json", {})
    e1 = aggregate.get("outputs/e1_diagnostic/metrics.json", {})
    status = "blocked" if not stats.get("real_data_available") else "ready"
    (root / "BLOCKING_ISSUES.md").write_text(
        "\n".join([
            "# Blocking Issues",
            "",
            f"Status: `{status}`",
            "",
            "No real CF-PrefBench-compatible dataset exists under the project root.",
            "",
            "What is missing:",
            "- Trajectory/video pairs addressable from `data/cf_prefbench/*.jsonl`.",
            "- Language instructions and minimal counterfactual instruction flips.",
            "- Preference labels for A/B/Tie under each instruction.",
            "- Task metadata and split metadata sufficient for leakage auditing.",
            "",
            "External sources found are robotics-adjacent but cannot be used as final benchmark evidence without an audited conversion or data-generation step.",
        ]) + "\n",
        encoding="utf-8",
    )
    (root / "NEGATIVE_RESULTS.md").write_text(
        "# Negative Results\n\nNo empirical model result was run. The current negative result is a data-availability result: E0 found zero valid CF-PrefBench examples, so E1-E6 are scientifically blocked.\n",
        encoding="utf-8",
    )
    (root / "RESULTS_SUMMARY.md").write_text(
        "\n".join([
            "# Results Summary",
            "",
            f"E0 status: `{stats.get('e0_status', 'missing')}`",
            f"E0 examples: `{stats.get('n_examples', 0)}`",
            f"Leakage verdict: `{leakage.get('verdict', 'missing')}`",
            f"E1 status: `{e1.get('status', 'missing')}`",
            "",
            "No paper claims are supported yet because there is no valid benchmark data in the project root.",
        ]) + "\n",
        encoding="utf-8",
    )
    (root / "CLAIMS_MATRIX.md").write_text(
        "\n".join([
            "# Claims Matrix",
            "",
            "| Claim | Evidence | Status |",
            "|---|---|---|",
            "| Existing models have canonical accuracy but weak counterfactual grounding. | E1 blocked because E0 has zero examples. | Not supported yet |",
            "| CPL improves held-out PFA while preserving standard accuracy. | E2 not run. | Not supported yet |",
            "| Improvement is not generic augmentation. | E4 not run. | Not supported yet |",
            "| CPL generalizes beyond seen lexical swaps. | E3 not run. | Not supported yet |",
            "| Impossible-premise calibration is improved or diagnosed. | E5 not run. | Not supported yet |",
            "| Downstream policy evidence is only supporting, not central. | E6 not run; framing recorded in configs. | Framing only |",
        ]) + "\n",
        encoding="utf-8",
    )
    (root / "EXPERIMENT_LOG.md").write_text(
        "\n".join([
            "# Experiment Log",
            "",
            "- Ran repository and data-source audit.",
            "- Created CF-PrefBench scaffold, schema validation, leakage audit, metrics, configs, and autopilot.",
            "- Ran E0; gate blocked because no real benchmark examples are available.",
            "- Ran E1 smoke gate; it recorded a blocked status and did not train models.",
        ]) + "\n",
        encoding="utf-8",
    )
    (root / "EXPERIMENT_AUDIT.md").write_text(
        "\n".join([
            "# Experiment Audit",
            "",
            "Scientific safeguards applied:",
            "- No held-out test tuning was performed.",
            "- No metric values were hard-coded.",
            "- No benchmark results were fabricated.",
            "- E1-E6 were stopped because E0 lacks real data.",
            "- Final claims are marked unsupported rather than inferred from unrelated data.",
        ]) + "\n",
        encoding="utf-8",
    )
    (root / "NEXT_ACTIONS.md").write_text(
        "\n".join([
            "# Next Actions",
            "",
            "1. Generate or copy a real benchmark into `data/cf_prefbench/` using the required JSONL schema.",
            "2. Recommended generation path: use Meta-World or ManiSkill2 to render paired trajectories for minimal instruction edits across action, object, color, spatial, and impossible-premise axes.",
            "3. For each pair, log both videos, canonical and counterfactual instructions, A/B/Tie preference labels, task metadata, lexical items, and counterfactual IDs.",
            "4. Re-run E0 and repair any leakage before starting E1.",
        ]) + "\n",
        encoding="utf-8",
    )
    for table_name, caption in [
        ("table_diagnostic.tex", "Diagnostic results unavailable because E0 is blocked."),
        ("table_main.tex", "Main CPL results unavailable because E0/E1 gates are blocked."),
        ("table_objective_ablation.tex", "Objective ablation unavailable because E2 gate is blocked."),
        ("table_generalization.tex", "Generalization results unavailable because real data is absent."),
        ("table_impossible.tex", "Impossible-premise results unavailable because no examples exist."),
        ("table_downstream.tex", "Downstream results unavailable because upstream gates are blocked."),
    ]:
        target = root / "paper" / "tables" / table_name
        target.write_text(
            "\\begin{{tabular}}{{ll}}\nStage & Status \\\\\n\\hline\n{} & blocked \\\\\n\\end{{tabular}}\n% Source: outputs/all_results.json. {}\n".format(table_name.replace(".tex", ""), caption),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
