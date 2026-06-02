#!/usr/bin/env python3
"""Regenerate main cliff figures (2, 3, 4) from RELEASED result files.

Writes PNGs to figures/reproduced/. These are independent re-renders from the
committed numbers (no retraining); the authoritative camera-ready figures live
in figures/main and figures/appendix, and the final paper is docs/EMNLP_final.pdf.
Figure 1 (LIB v0 architecture) is a hand-drawn diagram with no numeric source;
see docs/EMNLP_final.pdf page 3.
"""
from __future__ import annotations
import json, glob, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MET = os.path.join(ROOT, "results", "metrics")
TAB = os.path.join(ROOT, "results", "tables")
OUT = os.path.join(ROOT, "figures", "reproduced")
os.makedirs(OUT, exist_ok=True)

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:
    print(f"SKIP figure reproduction: matplotlib/numpy unavailable ({e}). "
          f"Authoritative figures are in figures/main and figures/appendix.")
    raise SystemExit(0)

def jload(p): return json.load(open(p))
def mean(xs): return sum(xs)/len(xs) if xs else float("nan")
CLIFF6 = ["motion_above","motion_below","size_BIG","size_SMALL","speed_FAST","speed_SLOW"]

def cell_means(glob_pat, only_seeds=None):
    files = sorted(glob.glob(os.path.join(MET, glob_pat)))
    if only_seeds: files = [f for f in files if any(f"seed{s}_" in os.path.basename(f) for s in only_seeds)]
    acc = {c: [] for c in CLIFF6}
    for fp in files:
        ca = jload(fp).get("class_aggregates", {})
        for c in CLIFF6:
            if c in ca: acc[c].append(ca[c]["accuracy_mean"])
    return {c: mean(v) for c, v in acc.items() if v}

# ---- Figure 2: 2x2 encoder factorial heatmap ----
cells = {"B-B": ("EXP-B_B-B_published_*", None), "B-L": ("EXP-B_B-L_*", ("42","123","2024")),
         "L-B": ("EXP-B_L-B_*", ("42","123","2024")), "L-L": ("EXP-B_L-L_published_*", None)}
vals = {k: mean(list(cell_means(p, s).values())) for k, (p, s) in cells.items()}
grid = np.array([[vals["B-B"], vals["B-L"]], [vals["L-B"], vals["L-L"]]])
fig, ax = plt.subplots(figsize=(4.2, 3.6))
im = ax.imshow(grid, cmap="viridis", vmin=0.5, vmax=0.85)
ax.set_xticks([0,1], ["text=B/32","text=L/14"]); ax.set_yticks([0,1], ["vis=B/32","vis=L/14"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{grid[i,j]:.3f}", ha="center", va="center", color="w", fontsize=12)
ax.set_title("Figure 2 (repro): 2x2 encoder factorial\ncliff accuracy")
fig.colorbar(im, fraction=0.046); fig.tight_layout()
fig.savefig(os.path.join(OUT, "figure2_2x2_factorial_reproduced.png"), dpi=150); plt.close(fig)

# ---- Figure 3: cross-family per-class ----
fam = {"B-B": ("EXP-B_B-B_published_*", None), "OC-OC": ("EXP-H_OC-OC_*", None), "SG-SG": ("EXP-H_SG-SG_*", None)}
fm = {k: cell_means(p, s) for k, (p, s) in fam.items()}
x = np.arange(len(CLIFF6)); w = 0.26
fig, ax = plt.subplots(figsize=(8, 3.8))
for i, (k, d) in enumerate(fm.items()):
    ax.bar(x + (i-1)*w, [d.get(c, 0) for c in CLIFF6], w, label=k)
ax.axhline(0.5, ls="--", c="gray", lw=1)
ax.set_xticks(x, [c.replace("_","\n") for c in CLIFF6], fontsize=8)
ax.set_ylabel("3-seed mean accuracy"); ax.set_ylim(0, 1.05); ax.legend()
ax.set_title("Figure 3 (repro): cross-family validation")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figure3_cross_family_reproduced.png"), dpi=150); plt.close(fig)

# ---- Figure 4(b): size/speed class-bimodal ----
data = jload(os.path.join(TAB, "table3_4_5_7_18_cliff_table.json"))["data"]
agg = {}
for d in data:
    ax_name = "Size" if d["token"].startswith("size_") else "Speed"
    key = f"{ax_name}\n{d['cls']}"
    agg.setdefault(key, {"b32": [], "l14": []})
    agg[key]["b32"].append(d["b32_mean"]); agg[key]["l14"].append(d["l14_mean"])
keys = list(agg.keys()); xx = np.arange(len(keys))
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.bar(xx-0.2, [mean(agg[k]["b32"]) for k in keys], 0.4, label="ViT-B/32")
ax.bar(xx+0.2, [mean(agg[k]["l14"]) for k in keys], 0.4, label="ViT-L/14")
ax.axhline(0.5, ls="--", c="gray", lw=1)
ax.set_xticks(xx, keys, fontsize=8); ax.set_ylim(0, 1.05); ax.set_ylabel("accuracy (n=8 pool)")
ax.set_title("Figure 4(b) (repro): size/speed class-bimodal cliff"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figure4_size_speed_reproduced.png"), dpi=150); plt.close(fig)

print("Wrote reproduced figures to", OUT)
for f in sorted(os.listdir(OUT)): print("  -", f)
