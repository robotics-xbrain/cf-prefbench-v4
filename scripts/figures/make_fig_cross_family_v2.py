"""Figure 8 — fig_cross_family_v2 (camera-ready edition).

SigLIP cross-family validation: 6 cliff classes × 3 contrastive systems
(B-B / OC-OC / SG-SG), n=3 seeds per system.

Differences from the figures_new/ v1 + v2:
  1. Legend moved above plot, 6 entries in one row
     (bbox_to_anchor=(0.5, 1.15), ncol=6, frameon=False, fontsize=8)
     figsize bumped to (9, 4.8) for vertical headroom (was 4.5 in
     figures_new/fig_cross_family_v2.pdf)
  2. Legend includes 3 custom handles: cliff hatch swatch,
     closure border swatch, chance line
  3. hatch.linewidth = 0.5 locally (red hatch no longer overpowers
     gray B-B bar color)

Data: aggregated from per-seed cliff_tokens_seed*.json (3 cells × 3 seeds).
Validated against the figures_new/fig_cross_family.pdf reference values
within 0.005 tolerance.

Output: figures_camera_ready/fig_cross_family_v2.pdf + .png
"""
from __future__ import annotations
import json, glob, sys, subprocess
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path("/path/to/project")
sys.path.insert(0, str(ROOT / ".skills/nature-figure-style"))
from style import apply_nature_style, PALETTE
from helpers import add_chance_line

OUT_DIR = ROOT / "figures_camera_ready"
OUT_PDF = OUT_DIR / "fig_cross_family_v2.pdf"
OUT_PNG = OUT_DIR / "fig_cross_family_v2.png"

CLASS_ORDER = ("motion_above", "motion_below", "size_BIG", "size_SMALL",
               "speed_FAST", "speed_SLOW")
DISPLAY_TWO_LINE = {
    "motion_above": "Motion\nabove",
    "motion_below": "Motion\nbelow",
    "size_BIG":     "Size BIG",
    "size_SMALL":   "Size SMALL",
    "speed_FAST":   "Speed FAST",
    "speed_SLOW":   "Speed SLOW",
}
EXPECTED = {
    "B-B":   {"motion_above": 0.887, "motion_below": 0.595,
              "size_BIG": 0.925, "size_SMALL": 0.444,
              "speed_FAST": 0.115, "speed_SLOW": 0.829},
    "OC-OC": {"motion_above": 0.679, "motion_below": 0.839,
              "size_BIG": 0.972, "size_SMALL": 0.929,
              "speed_FAST": 0.321, "speed_SLOW": 0.639},
    "SG-SG": {"motion_above": 0.661, "motion_below": 0.780,
              "size_BIG": 0.929, "size_SMALL": 0.817,
              "speed_FAST": 0.619, "speed_SLOW": 0.667},
}
CELL_GLOBS = {
    "B-B":   str(ROOT / "experiments/EXP-B/B-B_published/seed*/cliff_tokens_seed*.json"),
    "OC-OC": str(ROOT / "experiments/EXP-H/OC-OC/seed*/cliff_tokens_seed*.json"),
    "SG-SG": str(ROOT / "experiments/EXP-H/SG-SG/seed*/cliff_tokens_seed*.json"),
}


def load_cell(cell: str):
    """Returns dict[class -> (mean, std)] aggregated over 3 seeds."""
    paths = sorted(glob.glob(CELL_GLOBS[cell]))
    if len(paths) != 3:
        raise FileNotFoundError(f"{cell}: expected 3 JSONs, found {len(paths)}")
    dicts = [json.load(open(p)) for p in paths]
    out = {}
    for cls in CLASS_ORDER:
        vals = [d["class_aggregates"][cls]["accuracy_mean"] for d in dicts]
        out[cls] = (float(np.mean(vals)), float(np.std(vals, ddof=0)))
    return out


def verify_render(pdf_path, png_path):
    import re
    assert pdf_path.stat().st_size > 5 * 1024
    assert png_path.stat().st_size > 30 * 1024
    info = subprocess.check_output(["pdfinfo", str(pdf_path)], text=True)
    m_pages = re.search(r"^Pages:\s+(\d+)", info, re.M)
    assert m_pages and int(m_pages.group(1)) == 1
    m_size = re.search(r"^Page size:\s+([\d.]+)\s+x\s+([\d.]+)\s+pts", info, re.M)
    assert m_size
    w, h = float(m_size.group(1)), float(m_size.group(2))
    assert w > 100 and h > 100
    im = Image.open(png_path)
    assert im.size[0] > 300 and im.size[1] > 150
    print(f"  RENDER OK  pdf={pdf_path.stat().st_size}B  pdf_size={w:.0f}x{h:.0f}pt  png={png_path.stat().st_size}B  png_px={im.size}")


def main():
    apply_nature_style()
    mpl.rcParams["hatch.linewidth"] = 0.5  # FIX 3 — thinner red hatch

    # --- Load and validate ---
    aggregates = {c: load_cell(c) for c in ("B-B", "OC-OC", "SG-SG")}
    print("=" * 78)
    print("VALIDATION: per-cell mean vs expected (tolerance < 0.005)")
    print("=" * 78)
    print(f"{'cell':5s} {'class':14s} {'mean':>8s} {'std':>8s} {'exp':>8s} {'Δ':>9s}")
    bad = []
    for cell in ("B-B", "OC-OC", "SG-SG"):
        for cls in CLASS_ORDER:
            m, s = aggregates[cell][cls]
            exp = EXPECTED[cell][cls]
            d = m - exp
            if abs(d) >= 0.005: bad.append((cell, cls, d))
            print(f"  {cell:4s} {cls:14s} {m:>8.4f} {s:>8.4f} {exp:>8.3f} {d:>+9.4f}")
    if bad:
        raise RuntimeError(f"validation failed: {bad}")

    means = {cell: [aggregates[cell][c][0] for c in CLASS_ORDER]
             for cell in ("B-B", "OC-OC", "SG-SG")}
    stds  = {cell: [aggregates[cell][c][1] for c in CLASS_ORDER]
             for cell in ("B-B", "OC-OC", "SG-SG")}
    class_labels = [DISPLAY_TWO_LINE[c] for c in CLASS_ORDER]

    # --- Plot ---
    n_cls = 6
    w = 0.25
    x = np.arange(n_cls)

    # FIX 1 — taller figsize to accommodate header legend
    fig, ax = plt.subplots(figsize=(9, 4.8))

    bars_bb = ax.bar(x - w, means["B-B"], w, yerr=stds["B-B"],
                     color=PALETTE["b_b"], edgecolor="white", linewidth=0.5,
                     capsize=3, ecolor="black",
                     error_kw=dict(elinewidth=0.8))
    bars_oc = ax.bar(x,     means["OC-OC"], w, yerr=stds["OC-OC"],
                     color=PALETTE["oc_oc"], edgecolor="white", linewidth=0.5,
                     capsize=3, ecolor="black",
                     error_kw=dict(elinewidth=0.8))
    bars_sg = ax.bar(x + w, means["SG-SG"], w, yerr=stds["SG-SG"],
                     color=PALETTE["sg_sg"], edgecolor="white", linewidth=0.5,
                     capsize=3, ecolor="black",
                     error_kw=dict(elinewidth=0.8))

    # Cliff + closure highlights
    n_cliff = 0
    n_closure = 0
    for i in range(n_cls):
        if means["B-B"][i] <= 0.5:
            bars_bb[i].set_edgecolor(PALETTE["cliff"])
            bars_bb[i].set_linewidth(2.5)
            bars_bb[i].set_hatch("//")
            n_cliff += 1
            if means["OC-OC"][i] > 0.7:
                bars_oc[i].set_edgecolor(PALETTE["closure"])
                bars_oc[i].set_linewidth(2.5)
                n_closure += 1
            if means["SG-SG"][i] > 0.7:
                bars_sg[i].set_edgecolor(PALETTE["closure"])
                bars_sg[i].set_linewidth(2.5)
                n_closure += 1

    add_chance_line(ax, y=0.5, label=None)  # explicit legend entry below
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels)
    ax.set_ylabel("3-seed mean accuracy")
    ax.set_ylim(0.0, 1.0)

    # FIX 2 — header legend with 6 custom handles
    handles = [
        Patch(facecolor=PALETTE["b_b"],   edgecolor="white", linewidth=0.5,
              label="B-B (OpenAI CLIP)"),
        Patch(facecolor=PALETTE["oc_oc"], edgecolor="white", linewidth=0.5,
              label="OC-OC (LAION-2B)"),
        Patch(facecolor=PALETTE["sg_sg"], edgecolor="white", linewidth=0.5,
              label="SG-SG (SigLIP)"),
        Patch(facecolor="lightgray", edgecolor=PALETTE["cliff"],
              linewidth=2.5, hatch="//",
              label=r"B-B cliff ($\leq 0.5$)"),
        Patch(facecolor="lightgray", edgecolor=PALETTE["closure"],
              linewidth=2.5, label=r"cliff closure ($> 0.7$)"),
        Line2D([0], [0], color=PALETTE["chance"], linestyle="--",
               linewidth=1.0, label="chance"),
    ]
    leg = ax.legend(handles=handles,
                    loc="upper center", bbox_to_anchor=(0.5, 1.15),
                    ncol=6, frameon=False, fontsize=8,
                    handletextpad=0.5, columnspacing=1.2)

    # --- Assertions ---
    assert len(bars_bb) == 6 and len(bars_oc) == 6 and len(bars_sg) == 6
    assert all(b.get_height() > 0 for b in bars_bb)
    assert all(b.get_height() > 0 for b in bars_oc)
    assert all(b.get_height() > 0 for b in bars_sg)
    assert len(leg.get_texts()) == 6, f"legend items: {len(leg.get_texts())}"

    fig.savefig(OUT_PDF, bbox_inches="tight", dpi=300)
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=300)
    plt.close(fig)

    # Report
    print("\n## Bar heights (sanity)")
    print(f"B-B   : {[round(b.get_height(), 3) for b in bars_bb]}")
    print(f"OC-OC : {[round(b.get_height(), 3) for b in bars_oc]}")
    print(f"SG-SG : {[round(b.get_height(), 3) for b in bars_sg]}")
    print(f"\nHighlights — cliff: {n_cliff}, closure: {n_closure}")
    print(f"Legend items: {len(leg.get_texts())}")
    print(f"figsize: (9, 4.8) — bumped for header legend")
    print(f"hatch.linewidth: 0.5 (override after apply_nature_style)")
    print()
    verify_render(OUT_PDF, OUT_PNG)
    print()
    print(f"PDF: {OUT_PDF.resolve()}")
    print(f"PNG: {OUT_PNG.resolve()}")


if __name__ == "__main__":
    main()
