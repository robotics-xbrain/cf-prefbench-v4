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

# hygiene: allow top-level .git in a normal git clone, but reject nested VCS/cache/log artifacts.
# Reviewer workflow usually runs verify_artifact.sh from a cloned repository, where ROOT/.git is expected.
from pathlib import Path as _Path

ROOT_PATH = _Path(ROOT)
bad = []
ignored_top_level = {".git", ".venv", "venv", "env", ".tox", ".pytest_cache", ".mypy_cache"}

def is_ignored(path):
    try:
        rel = path.relative_to(ROOT_PATH)
    except ValueError:
        return False
    parts = rel.parts
    return bool(parts and parts[0] in ignored_top_level)

# Top-level .git is allowed; nested .git directories are not.
for x in ROOT_PATH.rglob(".git"):
    if is_ignored(x):
        continue
    bad.append(str(x.relative_to(ROOT_PATH)))

for pat in ["__pycache__", "wandb", "runs", "logs", "prompts", "review-stage", "research-wiki"]:
    for x in ROOT_PATH.rglob(pat):
        if is_ignored(x):
            continue
        bad.append(str(x.relative_to(ROOT_PATH)))

for pat in ["*.pyc", "texput.log"]:
    for x in ROOT_PATH.rglob(pat):
        if is_ignored(x):
            continue
        bad.append(str(x.relative_to(ROOT_PATH)))

chk("hygiene:no nested vcs/cache/logs", not bad, f"offenders={bad}" if bad else "clean")

# identity scan ran
chk("identity_scan_present", os.path.exists(os.path.join(ROOT, "docs/IDENTITY_LEAK_SCAN.md")), "")
# smoke test result present
chk("smoke_test_result_present", os.path.exists(os.path.join(ROOT, "smoke_test_results.json")), "")

overall = all(c["ok"] for c in checks)
res = {"root": ROOT, "overall": "PASS" if overall else "FAIL", "checks": checks}
json.dump(res, open(os.path.join(ROOT, "verify_artifact_results.json"), "w"), indent=2)
print(f"\nOVERALL: {res['overall']}")
raise SystemExit(0 if overall else 1)
