"""Figure 7 — fig_4_5_cross_method_v2.

Cross-method cliff comparison: 3 panels (Motion / Size / Speed), each
with 4 tokens × 4 methods grouped bars. 12 tokens total, 48 cells.

DATA dict is read from scripts/make_fig_4_5_cross_method.py (the same
literals that produced v1), via top-of-file exec — avoids running the
original main() while binding us to the source.

Differences from v1:
  - 4-color restrained palette:
      LIB-B/32:  PALETTE["vit_b32"]      (blue)
      LIB-L/14:  PALETTE["vit_l14"]      (orange)
      GPT-4o:    PALETTE["closure"]      (green — narrative: closes cliff)
      Qwen-2B:   PALETTE["muted"]        (gray — near chance, secondary)
  - Cliff hatching only on LIB-B/32 bars where is_cliff_for_B32 is True
    (the 6 known cliff tokens — assertion at the bottom)
  - Star marker (best-of-4 on each cliff token) repositioned just above
    the bar, smaller (s≈70, PALETTE["cliff"] red)
  - Legend single horizontal row above 3 panels, no frame
  - x-tick rotation 30° (was 45°)
  - Panel titles "(a) Motion" / "(b) Size" / "(c) Speed"

Output: figures_camera_ready/fig_4_5_cross_method_v2.pdf + .png
"""
from __future__ import annotations
import sys
import re
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
OUT_PDF = OUT_DIR / "fig_4_5_cross_method_v2.pdf"
OUT_PNG = OUT_DIR / "fig_4_5_cross_method_v2.png"

# Color assignment per spec
METHOD_COLORS = {
    "LIB-B/32":  PALETTE["vit_b32"],
    "LIB-L/14":  PALETTE["vit_l14"],
    "GPT-4o":    PALETTE["closure"],
    "Qwen-2B":   PALETTE["muted"],
}
METHODS = ["LIB-B/32", "LIB-L/14", "GPT-4o", "Qwen-2B"]
EXPECTED_CLIFF_TOKENS = {"scoot", "transit", "miniature", "petite",
                          "briskly", "speedily"}

PANEL_TITLES = {"motion": "(a) Motion", "size": "(b) Size",
                "speed": "(c) Speed"}


def load_source_data():
    """Exec the top of scripts/make_fig_4_5_cross_method.py to get DATA."""
    src_path = ROOT / "scripts/make_fig_4_5_cross_method.py"
    src_text = src_path.read_text()
    m = re.search(r"^def main\(\):", src_text, re.M)
    if m is None:
        raise RuntimeError("could not find 'def main():' in source")
    ns = {}
    exec(src_text[:m.start()], ns)
    return ns["DATA"]


def verify_render(pdf_path, png_path):
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

    DATA = load_source_data()

    # --- Validate the loaded DATA against expected structure ---
    actual_cliff = set()
    for axis, rows in DATA.items():
        for tok, is_cliff, b32, l14, gpt, qwen in rows:
            if is_cliff:
                actual_cliff.add(tok)
            # all 4 method values must be in [0, 1]
            for v in (b32, l14, gpt, qwen):
                assert 0.0 <= v <= 1.0, f"value out of [0,1]: {tok} {v}"
    assert actual_cliff == EXPECTED_CLIFF_TOKENS, \
        f"cliff token mismatch: {actual_cliff} vs {EXPECTED_CLIFF_TOKENS}"
    print(f"DATA OK — 6 cliff tokens: {sorted(actual_cliff)}")
    # 48 cells = 12 tokens × 4 methods
    n_cells = sum(4 for axis, rows in DATA.items() for _ in rows)
    assert n_cells == 48, f"got {n_cells} cells, expected 48"

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True,
                             constrained_layout=True)
    width = 0.20

    # Track all bars created for assertions
    cliff_hatched_count = 0
    non_cliff_b32_count = 0

    for ax_idx, axis_name in enumerate(("motion", "size", "speed")):
        ax = axes[ax_idx]
        rows = DATA[axis_name]
        tokens  = [r[0] for r in rows]
        cliffs  = [r[1] for r in rows]
        b32_v   = [r[2] for r in rows]
        l14_v   = [r[3] for r in rows]
        gpt_v   = [r[4] for r in rows]
        qwen_v  = [r[5] for r in rows]
        method_data = {"LIB-B/32": b32_v, "LIB-L/14": l14_v,
                       "GPT-4o": gpt_v,   "Qwen-2B": qwen_v}

        x = np.arange(len(tokens))
        offsets = {m: (i - 1.5) * width for i, m in enumerate(METHODS)}

        bars_by_method = {}
        for m in METHODS:
            bars = ax.bar(x + offsets[m], method_data[m], width,
                          color=METHOD_COLORS[m], edgecolor="white",
                          linewidth=0.4, zorder=2)
            bars_by_method[m] = list(bars)

        # Hatch LIB-B/32 cliff bars
        for i, is_cliff in enumerate(cliffs):
            if is_cliff:
                b = bars_by_method["LIB-B/32"][i]
                b.set_hatch("//")
                b.set_edgecolor(PALETTE["cliff"])
                b.set_linewidth(2.5)
                cliff_hatched_count += 1
            else:
                non_cliff_b32_count += 1

        # Star markers above the best-of-4 method on each cliff token
        for i, is_cliff in enumerate(cliffs):
            if not is_cliff:
                continue
            vals = [(m, method_data[m][i]) for m in METHODS]
            best_m, best_v = max(vals, key=lambda kv: kv[1])
            ax.scatter(x[i] + offsets[best_m], best_v + 0.04,
                       marker="*", s=70, color=PALETTE["cliff"],
                       edgecolors="white", linewidths=0.5, zorder=5)

        add_chance_line(ax, y=0.5,
                        label=("chance" if ax_idx == 0 else None),
                        label_pos="left")
        ax.set_xticks(x)
        ax.set_xticklabels(tokens, rotation=30, ha="right", fontsize=8.5)
        ax.set_ylim(0.0, 1.15)
        ax.set_title(PANEL_TITLES[axis_name], fontsize=11, fontweight="bold")
        if ax_idx == 0:
            ax.set_ylabel("Held-out row accuracy")

    # Shared legend above all 3 panels, single row, no frame
    legend_handles = [
        Patch(facecolor=PALETTE["vit_b32"], edgecolor="white", linewidth=0.4,
              label="LIB-B/32"),
        Patch(facecolor=PALETTE["vit_l14"], edgecolor="white", linewidth=0.4,
              label="LIB-L/14"),
        Patch(facecolor=PALETTE["closure"], edgecolor="white", linewidth=0.4,
              label="GPT-4o"),
        Patch(facecolor=PALETTE["muted"], edgecolor="white", linewidth=0.4,
              label="Qwen-2B"),
        Patch(facecolor="lightgray", edgecolor=PALETTE["cliff"],
              linewidth=2.5, hatch="//", label="LIB-B/32 cliff token"),
        Line2D([0], [0], marker="*", color="none",
               markerfacecolor=PALETTE["cliff"], markeredgecolor="white",
               markersize=10, label="best method (cliff)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.05), ncol=6, frameon=False,
               fontsize=8, handletextpad=0.5, columnspacing=1.4)

    # Pre-save assertions
    assert cliff_hatched_count == 6, f"got {cliff_hatched_count} hatched, expected 6"
    assert non_cliff_b32_count == 6, f"got {non_cliff_b32_count} non-cliff B/32 bars, expected 6"

    fig.savefig(OUT_PDF, bbox_inches="tight", dpi=300)
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=300)
    plt.close(fig)

    # Report
    print("\n## 12 tokens × 4 methods (48 cells)")
    print(f"{'axis':6s} {'token':12s} {'cliff':6s} {'B/32':>8s} {'L/14':>8s} {'GPT-4o':>8s} {'Qwen-2B':>8s}  best")
    for axis_name in ("motion", "size", "speed"):
        for tok, is_cliff, b32, l14, gpt, qwen in DATA[axis_name]:
            best = max(zip(METHODS, (b32, l14, gpt, qwen)), key=lambda kv: kv[1])[0] if is_cliff else ""
            print(f"  {axis_name:5s} {tok:12s} {'YES' if is_cliff else 'no':6s} "
                  f"{b32:>8.3f} {l14:>8.3f} {gpt:>8.3f} {qwen:>8.3f}  {best}")
    print(f"\ncliff bars hatched: {cliff_hatched_count} / 6 expected")
    print(f"non-cliff B/32 bars: {non_cliff_b32_count} / 6 expected")
    print()
    verify_render(OUT_PDF, OUT_PNG)
    print()
    print(f"PDF: {OUT_PDF.resolve()}")
    print(f"PNG: {OUT_PNG.resolve()}")


if __name__ == "__main__":
    main()
