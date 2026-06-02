"""Figure 2 — fig_4_4_mechanism_diagram_v2.

Two cliff mechanisms side by side:
  (a) Motion: cosine-monotonic — 4 verbs over a sigmoid illustrative curve
  (b) Size/Speed: class-bimodal — 4 class-mean bars with n=8 Table-5 CIs

Output: figures_camera_ready/fig_4_4_mechanism_diagram_v2.pdf + .png
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
import subprocess

ROOT = Path("/path/to/project")
sys.path.insert(0, str(ROOT / ".skills/nature-figure-style"))
from style import apply_nature_style, PALETTE
from helpers import add_chance_line

OUT_DIR = ROOT / "figures_camera_ready"
OUT_PDF = OUT_DIR / "fig_4_4_mechanism_diagram_v2.pdf"
OUT_PNG = OUT_DIR / "fig_4_4_mechanism_diagram_v2.png"

# --- Source data (verbatim from prompt) ---
# Panel (a) Motion: (cos, acc) per verb, 3-seed means
MOTION = [
    ("shift",    0.968, 0.917),
    ("convey",   0.938, 0.929),
    ("scoot",    0.927, 0.679),
    ("transit",  0.915, 0.500),
]
# Panel (b) Size/Speed: (mean, CI_lo, CI_hi) per class, n=8 from Table 5
SIZE_SPEED = [
    ("Size BIG",    0.926, 0.923, 0.929,  False),  # not cliff
    ("Size SMALL",  0.543, 0.482, 0.595,  True),   # cliff
    ("Speed FAST",  0.092, 0.049, 0.150,  True),   # cliff
    ("Speed SLOW",  0.826, 0.757, 0.896,  False),  # not cliff
]


def panel_letter(ax, letter):
    ax.text(-0.08, 1.05, letter, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom", ha="left")


def verify_render(pdf_path: Path, png_path: Path):
    """Mandatory render verification."""
    import re
    assert pdf_path.stat().st_size > 5 * 1024, f"PDF too small: {pdf_path.stat().st_size}"
    assert png_path.stat().st_size > 30 * 1024, f"PNG too small: {png_path.stat().st_size}"
    info = subprocess.check_output(["pdfinfo", str(pdf_path)], text=True)
    m_pages = re.search(r"^Pages:\s+(\d+)", info, re.M)
    assert m_pages and int(m_pages.group(1)) == 1, f"page count != 1:\n{info}"
    m_size = re.search(r"^Page size:\s+([\d.]+)\s+x\s+([\d.]+)\s+pts", info, re.M)
    assert m_size, f"no page-size line in pdfinfo:\n{info}"
    w, h = float(m_size.group(1)), float(m_size.group(2))
    assert w > 100 and h > 100, f"PDF page too small: {w}x{h}"
    im = Image.open(png_path)
    assert im.size[0] > 300 and im.size[1] > 150, f"PNG too small: {im.size}"
    print(f"  RENDER OK  pdf={pdf_path.stat().st_size}B  pdf_size={w:.0f}x{h:.0f}pt  png={png_path.stat().st_size}B  png_px={im.size}")


def main():
    apply_nature_style()
    mpl.rcParams["hatch.linewidth"] = 0.5  # thinner red hatch (universal rule)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    # === Panel (a) Motion ===
    ax = axes[0]
    motion_x = np.array([v[1] for v in MOTION])
    motion_y = np.array([v[2] for v in MOTION])
    motion_names = [v[0] for v in MOTION]

    assert len(motion_x) == 4, f"len(motion_x)={len(motion_x)}"
    assert not np.isnan(motion_y).any(), "NaN in motion y values"

    # Background sigmoid illustrative curve
    x_curve = np.linspace(0.90, 0.97, 200)
    y_curve = 0.5 + 0.43 / (1 + np.exp(-50 * (x_curve - 0.93)))
    ax.plot(x_curve, y_curve, color=PALETTE["neutral"],
            linewidth=1.5, alpha=0.4, zorder=1)

    # Scatter — failing (transit, acc<=0.5) red diamond, passing blue circles
    fail = motion_y <= 0.5
    if fail.any():
        ax.scatter(motion_x[fail], motion_y[fail],
                   s=60, c=PALETTE["cliff"], marker="D",
                   edgecolors="white", linewidths=0.6, alpha=0.9, zorder=3)
    if (~fail).any():
        ax.scatter(motion_x[~fail], motion_y[~fail],
                   s=50, c=PALETTE["neutral"], marker="o",
                   edgecolors="white", linewidths=0.6, alpha=0.9, zorder=3)

    # Verb labels
    for n, x, y in zip(motion_names, motion_x, motion_y):
        ax.annotate(n, xy=(x, y), xytext=(6, 6),
                    textcoords="offset points", fontsize=9, color="black")

    add_chance_line(ax, y=0.5, label="chance", label_pos="left")
    ax.set_xlim(0.905, 0.978)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Held-out verb cosine to train")
    ax.set_ylabel("Held-out accuracy")
    ax.set_title("Motion: cosine-monotonic", fontsize=11)
    panel_letter(ax, "(a)")

    # === Panel (b) Size/Speed ===
    ax = axes[1]
    bar_names = [c[0] for c in SIZE_SPEED]
    bar_means = np.array([c[1] for c in SIZE_SPEED])
    bar_lo    = np.array([c[2] for c in SIZE_SPEED])
    bar_hi    = np.array([c[3] for c in SIZE_SPEED])
    is_cliff  = [c[4] for c in SIZE_SPEED]

    assert len(bar_means) == 4, f"len(bar_means)={len(bar_means)}"
    assert not np.isnan(bar_means).any(), "NaN in bar means"

    # Asymmetric error bars
    yerr_lo = bar_means - bar_lo
    yerr_hi = bar_hi - bar_means

    x_pos = np.arange(len(bar_names))
    bars = ax.bar(x_pos, bar_means, 0.6,
                  color=PALETTE["neutral"], edgecolor="white", linewidth=0.5,
                  zorder=2)
    ax.errorbar(x_pos, bar_means, yerr=[yerr_lo, yerr_hi],
                fmt="none", ecolor="black", elinewidth=0.8,
                capsize=3, capthick=0.8, zorder=3)
    # Apply cliff styling
    for b, cliff in zip(bars, is_cliff):
        if cliff:
            b.set_hatch("//")
            b.set_edgecolor(PALETTE["cliff"])
            b.set_linewidth(2.5)

    add_chance_line(ax, y=0.5, label="chance", label_pos="right")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bar_names)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Class")
    ax.set_ylabel("3-seed mean accuracy (n=8 CI)")
    ax.set_title("Size/Speed: class-bimodal", fontsize=11)
    panel_letter(ax, "(b)")

    fig.savefig(OUT_PDF, bbox_inches="tight", dpi=300)
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=300)
    plt.close(fig)

    # --- Reports ---
    print("=" * 78)
    print("DATA VALUES USED")
    print("=" * 78)
    print("Panel (a) Motion (verb, cos, acc):")
    for n, x, y in zip(motion_names, motion_x, motion_y):
        print(f"  {n:8s} cos={x:.3f}  acc={y:.3f}")
    print("Panel (b) Size/Speed (class, mean, CI):")
    for n, m, lo, hi, c in SIZE_SPEED:
        print(f"  {n:12s} mean={m:.3f}  CI=[{lo:.3f},{hi:.3f}]  cliff={c}")
    print()
    verify_render(OUT_PDF, OUT_PNG)
    print()
    print(f"PDF: {OUT_PDF.resolve()}")
    print(f"PNG: {OUT_PNG.resolve()}")


if __name__ == "__main__":
    main()
