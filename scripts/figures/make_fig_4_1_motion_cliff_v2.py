"""Figure 4 — fig_4_1_motion_cliff_b32_v2.

Single-panel motion-verb cliff on ViT-B/32. 4 verbs × 3 seeds, with the
3-seed means connected by a line and the per-seed dots overlaid at
alpha=0.3.

Differences from v1 (paper/figures/fig_4_1_motion_cliff_b32.pdf):
  - Grid removed (skill default)
  - Pearson r annotation moved to lower-right (was upper-left over data)
  - All mean markers use PALETTE["neutral"] (no special red for transit —
    narrative of this figure is cosine-monotonic, NOT cliff/no-cliff)
  - Seed-level dots at alpha=0.3, mean markers at alpha=1.0
  - Verb names via ax.annotate next to each mean marker
  - Liberation Sans via skill (no DejaVu Sans fallback)

Source: scripts/make_section4_figures.py:fig_4_1_motion_cliff_b32 (literals)

Output: figures_camera_ready/fig_4_1_motion_cliff_b32_v2.pdf + .png
"""
from __future__ import annotations
import sys
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path("/path/to/project")
sys.path.insert(0, str(ROOT / ".skills/nature-figure-style"))
from style import apply_nature_style, PALETTE
from helpers import add_chance_line

OUT_DIR = ROOT / "figures_camera_ready"
OUT_PDF = OUT_DIR / "fig_4_1_motion_cliff_b32_v2.pdf"
OUT_PNG = OUT_DIR / "fig_4_1_motion_cliff_b32_v2.png"

# --- Source data (verbatim from scripts/make_section4_figures.py) ---
# Per-verb cosines (ordered by increasing cos for monotone line)
VERBS = ["transit", "scoot", "convey", "shift"]
COS   = [0.9153,    0.9266,  0.9378,   0.9681]
# 3 seed accuracies per verb (exact fractions of 28)
SEEDS = {
    "transit": [14/28, 14/28, 14/28],
    "scoot":   [19/28, 17/28, 21/28],
    "convey":  [26/28, 26/28, 26/28],
    "shift":   [26/28, 26/28, 25/28],
}
# Annotation literals (recomputed from per-row predictions; preserved here
# to match the figure_4_1 caption in the paper)
PEARSON_R = 0.808
PEARSON_P = 0.0015
SPEARMAN_RHO = 0.880
SPEARMAN_P = 0.0002

# Validation targets (3-decimal precision)
EXPECTED_MEANS = {"shift": 0.917, "convey": 0.929,
                  "scoot": 0.679, "transit": 0.500}


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

    means = np.array([float(np.mean(SEEDS[v])) for v in VERBS])
    stds  = np.array([float(np.std(SEEDS[v], ddof=0)) for v in VERBS])

    # --- Validation ---
    print("=" * 78)
    print("VALIDATION (tolerance < 0.001)")
    print("=" * 78)
    bad = []
    for v, m in zip(VERBS, means):
        exp = EXPECTED_MEANS[v]
        d = m - exp
        ok = abs(d) < 0.001
        print(f"  {v:8s} mean={m:.6f}  exp={exp:.3f}  Δ={d:+.6f}  [{'OK' if ok else 'BAD'}]")
        if not ok: bad.append(v)
    if bad:
        raise RuntimeError(f"validation failed for {bad}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(5.5, 4))

    # Per-seed dots first (alpha=0.3 overlay underneath the means line)
    for v, c in zip(VERBS, COS):
        for s in SEEDS[v]:
            ax.scatter(c, s, color=PALETTE["neutral"], s=20,
                       alpha=0.3, zorder=2, edgecolors="none")

    # Mean ± std line connecting all 4 verbs
    ax.errorbar(COS, means, yerr=stds,
                fmt="o-", color=PALETTE["neutral"],
                markersize=8, markerfacecolor=PALETTE["neutral"],
                markeredgecolor="white", markeredgewidth=0.8,
                linewidth=1.5, capsize=3, capthick=0.8,
                ecolor="black", elinewidth=0.8,
                alpha=1.0, zorder=3, label="3-seed mean")

    # Verb labels via annotate
    for v, c, m in zip(VERBS, COS, means):
        ax.annotate(v, xy=(c, m), xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=9, color="black")

    # Chance line (use default skill helper; suppress label since lower-right
    # is occupied by the r annotation)
    add_chance_line(ax, y=0.5, label=None)

    # Pearson / Spearman annotation — lower-right
    text = (f"Pearson $r={PEARSON_R:.3f}$ ($p={PEARSON_P:.4f}$)\n"
            f"Spearman $\\rho={SPEARMAN_RHO:.3f}$ ($p={SPEARMAN_P:.4f}$)")
    ax.text(0.97, 0.05, text, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color="black",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="none", alpha=0.85))

    # Axes
    ax.set_xlim(0.905, 0.978)
    ax.set_ylim(0.30, 1.05)
    ax.set_xlabel("CLIP-B/32 cosine to train")
    ax.set_ylabel("Held-out row accuracy")
    ax.set_title("Motion-verb cliff (ViT-B/32)")

    # Pre-save assertion
    assert len(means) == 4 and not np.isnan(means).any()

    fig.savefig(OUT_PDF, bbox_inches="tight", dpi=300)
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=300)
    plt.close(fig)

    # Report
    print("\n## scatter data (3-decimal precision)")
    print(f"{'verb':10s} {'cos':>7s} {'mean':>8s} {'std':>8s}  per-seed accs")
    for v, c, m, s in zip(VERBS, COS, means, stds):
        seeds_str = "[" + ", ".join(f"{a:.4f}" for a in SEEDS[v]) + "]"
        print(f"  {v:8s} {c:>7.4f} {m:>8.4f} {s:>8.4f}  {seeds_str}")
    print()
    verify_render(OUT_PDF, OUT_PNG)
    print()
    print(f"PDF: {OUT_PDF.resolve()}")
    print(f"PNG: {OUT_PNG.resolve()}")


if __name__ == "__main__":
    main()
