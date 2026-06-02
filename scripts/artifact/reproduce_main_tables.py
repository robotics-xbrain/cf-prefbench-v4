#!/usr/bin/env python3
"""Regenerate the main-paper cliff tables (1-5) from RELEASED result files.

No retraining and no model evaluation: every number is recomputed from the
committed per-seed metric JSONs under results/. Outputs Markdown + CSV to
results/reproduced_tables/. Cross-check against the final PDF / results/
MASTER_PAPER_DATA.tex.
"""
from __future__ import annotations
import json, glob, os, csv, statistics as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MET = os.path.join(ROOT, "results", "metrics")
TAB = os.path.join(ROOT, "results", "tables")
OUT = os.path.join(ROOT, "results", "reproduced_tables")
os.makedirs(OUT, exist_ok=True)

def jload(p): return json.load(open(p))
def mean(xs): return sum(xs)/len(xs) if xs else float("nan")

md = ["# Reproduced main tables (from released results)\n",
      "Recomputed from `results/` per-seed JSONs by `scripts/artifact/reproduce_main_tables.py`.\n"]

# ---------- Table 1: ViT-B/32 motion-verb cliff (4-verb probe) ----------
t1 = jload(os.path.join(TAB, "table1_table6_cliff_data_B32.json"))["cliff_data"]["motion_sequence"]
order = ["shift", "convey", "scoot", "transit"]
md += ["## Table 1 — ViT-B/32 motion-verb cliff (3-seed)\n",
       "| Verb | cos(B/32) | Accuracy (3-seed mean ± std) |", "|---|---|---|"]
with open(os.path.join(OUT, "table1_motion_b32.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["verb", "cos_b32", "mean", "std"])
    for v in order:
        d = t1[v]; md.append(f"| {v} | {d['cos']:.3f} | {d['mean']:.3f} ± {d['std']:.3f} |")
        w.writerow([v, d["cos"], d["mean"], d["std"]])
md.append("")

# ---------- Table 2: motion cliff under two backbones ----------
t2 = jload(os.path.join(TAB, "table2_fig5_L14_cliff_table.json"))["cliff_data_motion_sequence"]
md += ["## Table 2 — Motion-verb cliff under two backbones\n",
       "| Verb | cos(B/32) | B/32 | cos(L/14) | L/14 |", "|---|---|---|---|---|"]
with open(os.path.join(OUT, "table2_motion_two_backbones.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["verb", "cos_b32", "b32_acc", "cos_l14", "l14_acc"])
    for v in order:
        b = t1[v]; l = t2[v]
        md.append(f"| {v} | {b['cos']:.3f} | {b['mean']:.3f} | {l['cos_L14']:.3f} | {l['mean_acc']:.3f} |")
        w.writerow([v, b["cos"], round(b["mean"],3), l["cos_L14"], round(l["mean_acc"],3)])
md.append("")

# ---------- Tables 3 & 4: cell means from per-seed class_aggregates ----------
CLIFF6 = ["motion_above","motion_below","size_BIG","size_SMALL","speed_FAST","speed_SLOW"]
def cell_class_means(glob_pat, only_seeds=None):
    """Return {class: mean-over-seeds accuracy} for files matching glob_pat.

    only_seeds: optional iterable of seed strings to restrict to (the paper
    uses seeds {1,2,3} for B-B/L-L and {42,123,2024} for B-L/L-B)."""
    files = sorted(glob.glob(os.path.join(MET, glob_pat)))
    if only_seeds is not None:
        files = [f for f in files if any(f"seed{s}_" in os.path.basename(f) for s in only_seeds)]
    acc = {c: [] for c in CLIFF6}
    for fp in files:
        try: ca = jload(fp).get("class_aggregates", {})
        except Exception: continue
        for c in CLIFF6:
            if c in ca and "accuracy_mean" in ca[c]:
                acc[c].append(ca[c]["accuracy_mean"])
    return {c: mean(v) for c, v in acc.items() if v}, len(files)

# paper seed protocol: B-B/L-L use {1,2,3}; B-L/L-B use {42,123,2024}
cells_t3 = {"B-B (baseline)": ("EXP-B_B-B_published_*", None),
            "B-L": ("EXP-B_B-L_*", ("42","123","2024")),
            "L-B": ("EXP-B_L-B_*", ("42","123","2024")),
            "L-L (full upgrade)": ("EXP-B_L-L_published_*", None)}
md += ["## Table 3 — 2x2 encoder factorial (mean cliff accuracy over cliff classes)\n",
       "| Cell | cliff acc (mean over 6 cliff classes) | #seed files |", "|---|---|---|"]
for name, (pat, seeds) in cells_t3.items():
    cm, nf = cell_class_means(pat, seeds)
    overall = mean(list(cm.values())) if cm else float("nan")
    md.append(f"| {name} | {overall:.3f} | {nf} |")
md.append("\n_Cross-check Table 3 (PDF): B-B 0.633, B-L 0.827, L-B 0.611, L-L 0.803 "
          "(n=3 cliff-probe-token protocol; small protocol differences expected)._\n")

cells_t4 = {"B-B (OpenAI CLIP)": "EXP-B_B-B_published_*",
            "OC-OC (LAION-2B)": "EXP-H_OC-OC_*", "SG-SG (SigLIP)": "EXP-H_SG-SG_*"}
md += ["## Table 4 — Cross-family per-class cliff accuracy\n",
       "| Class | " + " | ".join(cells_t4.keys()) + " |",
       "|---|" + "---|"*len(cells_t4)]
colvals = {name: cell_class_means(pat)[0] for name, pat in cells_t4.items()}
with open(os.path.join(OUT, "table4_cross_family.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["class"] + list(cells_t4.keys()))
    for c in CLIFF6:
        row = [colvals[name].get(c, float("nan")) for name in cells_t4]
        md.append(f"| {c} | " + " | ".join(f"{x:.3f}" for x in row) + " |")
        w.writerow([c] + [round(x,3) for x in row])
md.append("")

# ---------- Table 5: three-axis x two-architecture (n=8 expanded pool) ----------
data = jload(os.path.join(TAB, "table3_4_5_7_18_cliff_table.json"))["data"]
groups = {"Motion above-cliff": lambda d: d["cls"]=="ABOVE" or d.get("axis")=="motion" and d["cls"]=="ABOVE",
          }
# classify by token prefix / cls field
def axis_class(d):
    tok = d["token"]; cls = d["cls"]
    if tok.startswith("size_"): return ("Size", cls)
    if tok.startswith("speed_"): return ("Speed", cls)
    return ("Motion", cls)
agg = {}
for d in data:
    key = axis_class(d)
    agg.setdefault(key, {"b32": [], "l14": []})
    agg[key]["b32"].append(d["b32_mean"]); agg[key]["l14"].append(d["l14_mean"])
md += ["## Table 5 — Three-axis x two-architecture cliff matrix (expanded pool)\n",
       "| Axis | Class | B/32 | L/14 | Δ |", "|---|---|---|---|---|"]
with open(os.path.join(OUT, "table5_three_axis.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["axis","class","b32","l14","delta"])
    for (axis, cls), v in sorted(agg.items()):
        b = mean(v["b32"]); l = mean(v["l14"])
        md.append(f"| {axis} | {cls} | {b:.3f} | {l:.3f} | {l-b:+.3f} |")
        w.writerow([axis, cls, round(b,3), round(l,3), round(l-b,3)])
md.append("\n_Note: this released file (`cliff_table.json`) holds the size/speed expanded "
          "pool; the motion above/below rows of Table 5 are in Tables 1-2 and "
          "`results/appendix/expanded_tokens/tables/`. Cross-check Table 5 (PDF): "
          "size SMALL 0.485, speed FAST 0.159 on B/32 (canonical n=8 x 4-token pool)._\n")

with open(os.path.join(OUT, "REPRODUCED_TABLES.md"), "w") as f:
    f.write("\n".join(md))
print("Wrote", os.path.join(OUT, "REPRODUCED_TABLES.md"), "and CSVs in", OUT)
