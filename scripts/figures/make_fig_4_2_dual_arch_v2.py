"""Figure 5 — fig_4_2_motion_dual_arch_v2.

Motion-verb cliff under two backbones (ViT-B/32 vs ViT-L/14), single
panel. 4 verbs × 2 backbones = 8 data points. Each backbone uses its
own cosine-to-train (architecture-specific x), so the two series are
plotted as separate lines.

Differences from v1:
  - B/32: PALETTE["vit_b32"] solid + circle markers (s=60)
  - L/14: PALETTE["vit_l14"] dashed + square markers (s=60)
  - B/32 verb labels offset above-right via connector line  (8, +8)
  - L/14 verb labels offset below-right via connector line  (8, -12)
    EXCEPT L/14 scoot — its cosine is 0.0015 from L/14 transit's, so
    its label is stacked further below (8, -28) to avoid collision
  - Skill rcParams (no grid, no top/right spine, Liberation Sans)
  - Legend top-right, no frame

Source: scripts/make_section4_figures.py:fig_4_2_motion_dual_arch (literals)

Output: figures_camera_ready/fig_4_2_motion_dual_arch_v2.pdf + .png
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
OUT_PDF = OUT_DIR / "fig_4_2_motion_dual_arch_v2.pdf"
OUT_PNG = OUT_DIR / "fig_4_2_motion_dual_arch_v2.png"

# --- Source data (verbatim from scripts/make_section4_figures.py) ---
VERBS   = ["transit", "scoot", "convey", "shift"]
B32_COS = [0.9153,    0.9266,  0.9378,   0.9681]
B32_ACC = [0.500,     0.679,   0.929,    0.917]
L14_COS = [0.8761,    0.8746,  0.9180,   0.9254]
L14_ACC = [0.917,     0.905,   0.929,    0.905]

# Per-verb label offset overrides for L/14 (default (8, -12); scoot
# moved further down to avoid colliding with transit at nearly the
# same cos).
L14_LABEL_OFFSETS = {
    "transit": (8, -12),
    "scoot":   (8, -28),  # 0.0015 away from transit in x → stack further down
    "convey":  (8, -12),
    "shift":   (8, -12),
}


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

    # --- Validation: 8 data points must match within 0.001 ---
    # Self-validation: data is hardcoded, so this is a no-op consistency check.
    assert len(VERBS) == 4
    assert len(B32_COS) == len(B32_ACC) == 4
    assert len(L14_COS) == len(L14_ACC) == 4
    for arr in (B32_COS, B32_ACC, L14_COS, L14_ACC):
        assert all(not np.isnan(v) for v in arr)

    fig, ax = plt.subplots(figsize=(6, 4))

    # Lines (no markers — markers via separate scatter for size control)
    ax.plot(B32_COS, B32_ACC, color=PALETTE["vit_b32"], linestyle="-",
            linewidth=1.5, zorder=2)
    ax.plot(L14_COS, L14_ACC, color=PALETTE["vit_l14"], linestyle="--",
            linewidth=1.5, zorder=2)

    # Markers at s=60 via scatter (uniform pre-pool size)
    ax.scatter(B32_COS, B32_ACC, s=60, c=PALETTE["vit_b32"], marker="o",
               edgecolors="white", linewidths=0.6, alpha=0.9,
               zorder=3, label="ViT-B/32")
    ax.scatter(L14_COS, L14_ACC, s=60, c=PALETTE["vit_l14"], marker="s",
               edgecolors="white", linewidths=0.6, alpha=0.9,
               zorder=3, label="ViT-L/14")

    # Verb labels — B/32 uniformly above-right, L/14 below-right
    # with scoot stacked further down to avoid the transit overlap.
    for v, x, y in zip(VERBS, B32_COS, B32_ACC):
        ax.annotate(v, xy=(x, y), xytext=(8, 8),
                    textcoords="offset points", fontsize=8.5,
                    color=PALETTE["vit_b32"], ha="left", va="bottom",
                    arrowprops=dict(arrowstyle="-", linewidth=0.5,
                                    color="gray"))
    for v, x, y in zip(VERBS, L14_COS, L14_ACC):
        off = L14_LABEL_OFFSETS[v]
        ax.annotate(v, xy=(x, y), xytext=off,
                    textcoords="offset points", fontsize=8.5,
                    color=PALETTE["vit_l14"], ha="left", va="top",
                    arrowprops=dict(arrowstyle="-", linewidth=0.5,
                                    color="gray"))

    # Chance line
    add_chance_line(ax, y=0.5, label="chance", label_pos="left")

    # Axes
    ax.set_xlim(0.860, 0.985)
    ax.set_ylim(0.40, 1.05)
    ax.set_xlabel("CLIP cosine to train (architecture-specific)")
    ax.set_ylabel("Held-out row accuracy")
    ax.set_title("Cliff under two backbones")
    ax.legend(loc="upper right", frameon=False)

    fig.savefig(OUT_PDF, bbox_inches="tight", dpi=300)
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=300)
    plt.close(fig)

    # Report
    print("=" * 78)
    print("DATA VALUES (3-decimal precision)")
    print("=" * 78)
    print(f"{'verb':10s} {'B/32 cos':>10s} {'B/32 acc':>10s}    {'L/14 cos':>10s} {'L/14 acc':>10s}")
    for v, bc, ba, lc, la in zip(VERBS, B32_COS, B32_ACC, L14_COS, L14_ACC):
        print(f"  {v:8s} {bc:>10.4f} {ba:>10.3f}    {lc:>10.4f} {la:>10.3f}")
    print()
    print(f"label-overlap diagnostic: L/14 scoot cos={L14_COS[1]:.4f}, "
          f"L/14 transit cos={L14_COS[0]:.4f} (Δ={L14_COS[0]-L14_COS[1]:+.4f})")
    print(f"  → L/14 scoot label uses xytext=(8, -28) instead of default (8, -12)")
    print()
    verify_render(OUT_PDF, OUT_PNG)
    print()
    print(f"PDF: {OUT_PDF.resolve()}")
    print(f"PNG: {OUT_PNG.resolve()}")


if __name__ == "__main__":
    main()
