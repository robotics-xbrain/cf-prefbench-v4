#!/usr/bin/env python3
"""Verify the artifact's structural integrity. Writes verify_artifact_results.json."""
from __future__ import annotations
import json, os, glob

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
checks = []
def chk(name, ok, detail=""):
    checks.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

# required top-level files
for f in ["README.md", "requirements.txt", "artifact_manifest.md", "artifact_manifest.json",
          "LICENSE_OR_USAGE.md", "FINAL_ARTIFACT_REPORT.md"]:
    chk(f"file:{f}", os.path.exists(os.path.join(ROOT, f)), f)

# required dirs
for d in ["data/cf_prefbench_v4", "cf_pref_learning/models", "configs", "checkpoints/lib_b32",
          "scripts/artifact", "results/tables", "figures/main", "docs"]:
    p = os.path.join(ROOT, d)
    n = len(glob.glob(os.path.join(p, "**", "*"), recursive=True)) if os.path.isdir(p) else 0
    chk(f"dir:{d}", os.path.isdir(p) and n > 0, f"{n} entries")

# key assets
chk("data:v4 splits", len(glob.glob(os.path.join(ROOT, "data/cf_prefbench_v4/*.jsonl"))) >= 8,
    f"{len(glob.glob(os.path.join(ROOT,'data/cf_prefbench_v4/*.jsonl')))} jsonl")
chk("checkpoints:>=40 .pt", len(glob.glob(os.path.join(ROOT, "checkpoints/**/*.pt"), recursive=True)) >= 40,
    f"{len(glob.glob(os.path.join(ROOT,'checkpoints/**/*.pt'),recursive=True))} .pt")
chk("results:MASTER_PAPER_DATA", os.path.exists(os.path.join(ROOT, "results/MASTER_PAPER_DATA.tex")), "")
chk("figures:final pdf", os.path.exists(os.path.join(ROOT, "docs/EMNLP_final.pdf")), "")
chk("docs:index+map+classification",
    all(os.path.exists(os.path.join(ROOT, "docs", x)) for x in
        ["PAPER_ITEM_INDEX.md", "PAPER_EVIDENCE_MAP.md", "REPO_FILE_CLASSIFICATION.md"]), "")

# hygiene: no .git / __pycache__ / logs / wandb
bad = []
for pat in [".git", "__pycache__", "wandb", "runs"]:
    if glob.glob(os.path.join(ROOT, "**", pat), recursive=True): bad.append(pat)
if glob.glob(os.path.join(ROOT, "**", "*.pyc"), recursive=True): bad.append("*.pyc")
logdirs = [d for d in glob.glob(os.path.join(ROOT, "**", "logs"), recursive=True) if os.path.isdir(d)]
if logdirs: bad.append("logs/")
chk("hygiene:no vcs/cache/logs", not bad, f"offenders={bad}" if bad else "clean")

# identity scan ran
chk("identity_scan_present", os.path.exists(os.path.join(ROOT, "docs/IDENTITY_LEAK_SCAN.md")), "")
# smoke test result present
chk("smoke_test_result_present", os.path.exists(os.path.join(ROOT, "smoke_test_results.json")), "")

overall = all(c["ok"] for c in checks)
res = {"root": ROOT, "overall": "PASS" if overall else "FAIL", "checks": checks}
json.dump(res, open(os.path.join(ROOT, "verify_artifact_results.json"), "w"), indent=2)
print(f"\nOVERALL: {res['overall']}")
raise SystemExit(0 if overall else 1)
