"""Figure 6 — fig_4_3_3axis_matrix_v2 (UPDATED TO n=8).

Grouped bar chart: 6 cliff classes × 2 backbones (B/32, L/14), n=8 seeds.
Cliff cells on B/32 (motion_below, size_SMALL, speed_FAST) are hatched
red per the paper's Section 4 narrative.

Data lineage:
  - 3-seed means are aggregated from 8 cliff_tokens_seed*.json files per
    backbone (3 published + 5 EXP-F = n=8). They reproduce Table 5
    means within 0.001 — see validation block.
  - 95% CI brackets are hardcoded from Table 5 (paper main text). The
    paper's bootstrap procedure (per-row resampling over 84 rows) differs
    from a per-seed bootstrap (8 seed-means), so re-deriving the CIs from
    these JSONs alone is not possible; using Table 5 verbatim is the
    correct camera-ready choice.

Output: figures_camera_ready/fig_4_3_3axis_matrix_v2.pdf + .png
"""
from __future__ import annotations
import json, glob
import sys
import subprocess
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
OUT_PDF = OUT_DIR / "fig_4_3_3axis_matrix_v2.pdf"
OUT_PNG = OUT_DIR / "fig_4_3_3axis_matrix_v2.png"

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
# Cliff cells on B/32 narrative (NOT auto-derived from a 0.5 threshold —
# motion_below=0.621 and size_SMALL=0.543 are above chance but are the
# "cliff" classes per §4 of the paper).
CLIFF_CLASSES = {"motion_below", "size_SMALL", "speed_FAST"}

# --- Hardcoded from Table 5 (paper main text, n=8 seeds) ---
# Format: class -> {backbone: (mean, CI_lo, CI_hi)}
TABLE5 = {
    "motion_above": {"B/32": (0.902, 0.888, 0.915),
                     "L/14": (0.922, 0.906, 0.935)},
    "motion_below": {"B/32": (0.621, 0.562, 0.688),
                     "L/14": (0.917, 0.906, 0.926)},
    "size_BIG":     {"B/32": (0.926, 0.923, 0.929),
                     "L/14": (0.878, 0.842, 0.914)},
    "size_SMALL":   {"B/32": (0.543, 0.482, 0.595),
                     "L/14": (0.860, 0.812, 0.897)},
    "speed_FAST":   {"B/32": (0.092, 0.049, 0.150),
                     "L/14": (0.333, 0.289, 0.402)},
    "speed_SLOW":   {"B/32": (0.826, 0.757, 0.896),
                     "L/14": (0.997, 0.993, 1.000)},
}

# n=8 seed pool sources (validation only)
JSON_GLOBS = {
    "B/32": [str(ROOT / "experiments/EXP-B/B-B_published/seed*/cliff_tokens_seed*.json"),
             str(ROOT / "experiments/EXP-F/B-32/seed*/cliff_tokens_seed*.json")],
    "L/14": [str(ROOT / "experiments/EXP-B/L-L_published/seed*/cliff_tokens_seed*.json"),
             str(ROOT / "experiments/EXP-F/L-14/seed*/cliff_tokens_seed*.json")],
}


def aggregate_from_json(backbone):
    paths = []
    for g in JSON_GLOBS[backbone]:
        paths.extend(sorted(glob.glob(g)))
    if len(paths) != 8:
        raise FileNotFoundError(f"{backbone}: expected 8 JSONs, found {len(paths)}")
    out = {}
    for c in CLASS_ORDER:
        vals = [json.load(open(p))["class_aggregates"][c]["accuracy_mean"]
                for p in paths]
        out[c] = float(np.mean(vals))
    return out


def verify_render(pdf_path: Path, png_path: Path):
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
    mpl.rcParams["hatch.linewidth"] = 0.5

    # --- Validate Table 5 means against JSON aggregation (< 0.005) ---
    print("=" * 78)
    print("VALIDATION: Table 5 mean vs JSON aggregation (n=8, tolerance < 0.005)")
    print("=" * 78)
    bad = []
    for bb in ("B/32", "L/14"):
        agg = aggregate_from_json(bb)
        for c in CLASS_ORDER:
            em = TABLE5[c][bb][0]
            jm = agg[c]
            d = em - jm
            ok = abs(d) < 0.005
            if not ok: bad.append((bb, c, em, jm, d))
            print(f"  {bb} {c:14s} Table5={em:.3f}  JSON={jm:.4f}  Δ={d:+.4f}  [{'OK' if ok else 'BAD'}]")
    if bad:
        raise RuntimeError(f"Table 5 means do not match JSON aggregation: {bad}")

    # --- Plot ---
    means_b32 = np.array([TABLE5[c]["B/32"][0] for c in CLASS_ORDER])
    lo_b32    = np.array([TABLE5[c]["B/32"][1] for c in CLASS_ORDER])
    hi_b32    = np.array([TABLE5[c]["B/32"][2] for c in CLASS_ORDER])
    means_l14 = np.array([TABLE5[c]["L/14"][0] for c in CLASS_ORDER])
    lo_l14    = np.array([TABLE5[c]["L/14"][1] for c in CLASS_ORDER])
    hi_l14    = np.array([TABLE5[c]["L/14"][2] for c in CLASS_ORDER])

    yerr_b32 = np.vstack([means_b32 - lo_b32, hi_b32 - means_b32])
    yerr_l14 = np.vstack([means_l14 - lo_l14, hi_l14 - means_l14])

    bar_w = 0.35
    x = np.arange(len(CLASS_ORDER))

    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    bars_b32 = ax.bar(x - bar_w/2, means_b32, bar_w,
                      color=PALETTE["vit_b32"], edgecolor="white",
                      linewidth=0.5, zorder=2)
    bars_l14 = ax.bar(x + bar_w/2, means_l14, bar_w,
                      color=PALETTE["vit_l14"], edgecolor="white",
                      linewidth=0.5, zorder=2)
    ax.errorbar(x - bar_w/2, means_b32, yerr=yerr_b32, fmt="none",
                ecolor="black", elinewidth=0.8, capsize=3,
                capthick=0.8, zorder=3)
    ax.errorbar(x + bar_w/2, means_l14, yerr=yerr_l14, fmt="none",
                ecolor="black", elinewidth=0.8, capsize=3,
                capthick=0.8, zorder=3)

    # Apply cliff styling — B/32 only at cliff classes
    cliff_marked = 0
    for i, c in enumerate(CLASS_ORDER):
        if c in CLIFF_CLASSES:
            bars_b32[i].set_hatch("//")
            bars_b32[i].set_edgecolor(PALETTE["cliff"])
            bars_b32[i].set_linewidth(2.5)
            cliff_marked += 1

    add_chance_line(ax, y=0.5, label=None)  # legend includes a dedicated entry

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_TWO_LINE[c] for c in CLASS_ORDER])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Held-out row accuracy (n=8 seeds)")
    ax.set_title("Three-axis × two-architecture cliff matrix (n=8 seeds)",
                 fontsize=11)

    # Legend — 4 entries: B/32, L/14, cliff hatch, chance
    handles = [
        Patch(facecolor=PALETTE["vit_b32"], edgecolor="white", linewidth=0.5,
              label="ViT-B/32"),
        Patch(facecolor=PALETTE["vit_l14"], edgecolor="white", linewidth=0.5,
              label="ViT-L/14"),
        Patch(facecolor=PALETTE["vit_l14"], edgecolor=PALETTE["cliff"],
              linewidth=2.5, hatch="//", label="B/32 cliff cell"),
        Line2D([0], [0], color=PALETTE["chance"], linestyle="--",
               linewidth=1.0, label="chance"),
    ]
    # Use a representative bar (not cliff) for the cliff hatch swatch so the
    # face color is unambiguous — set it to a neutral light gray
    handles[2] = Patch(facecolor="lightgray", edgecolor=PALETTE["cliff"],
                      linewidth=2.5, hatch="//", label="B/32 cliff cell")
    leg = ax.legend(handles=handles, loc="upper right", frameon=False,
                    fontsize=9, ncol=1)

    # Pre-save assertions
    assert len(bars_b32) == 6, f"len(bars_b32)={len(bars_b32)}"
    assert len(bars_l14) == 6, f"len(bars_l14)={len(bars_l14)}"
    assert all(b.get_height() > 0 for b in bars_b32)
    assert all(b.get_height() > 0 for b in bars_l14)
    assert cliff_marked == 3, f"cliff_marked={cliff_marked}, expected 3"

    fig.savefig(OUT_PDF, bbox_inches="tight", dpi=300)
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=300)
    plt.close(fig)

    # Report
    print("\n## bar heights (n=8 means)")
    print(f"  {'class':14s} {'B/32':>8s} {'L/14':>8s}  cliff?")
    for i, c in enumerate(CLASS_ORDER):
        print(f"  {c:14s} {means_b32[i]:>8.3f} {means_l14[i]:>8.3f}  "
              f"{'YES (B/32)' if c in CLIFF_CLASSES else ''}")
    print(f"\ncliff cells hatched: {cliff_marked} of 6 B/32 bars")
    print()
    verify_render(OUT_PDF, OUT_PNG)
    print()
    print(f"PDF: {OUT_PDF.resolve()}")
    print(f"PNG: {OUT_PNG.resolve()}")


if __name__ == "__main__":
    main()
