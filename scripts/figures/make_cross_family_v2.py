#!/usr/bin/env python3
# ============= FIGURE 2: make_cross_family_v2.py =============

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "b_b": "#7f7f7f",
    "oc_oc": "#1f77b4",
    "sg_sg": "#ff7f0e",
    "vit_b32": "#1f77b4",
    "vit_l14": "#ff7f0e",
    "cliff": "#d62728",
    "closure": "#2ca02c",
    "chance": "#888888",
    "neutral": "#1f77b4",
    "muted": "#7f7f7f",
}

CLASSES = [
    "Motion\nabove",
    "Motion\nbelow",
    "Size BIG",
    "Size SMALL",
    "Speed FAST",
    "Speed SLOW",
]

MEANS_BB = [0.887, 0.595, 0.925, 0.444, 0.115, 0.829]
MEANS_OCOC = [0.679, 0.839, 0.972, 0.929, 0.321, 0.639]
MEANS_SGSG = [0.661, 0.780, 0.929, 0.817, 0.619, 0.667]

STDS_BB = [0.04, 0.06, 0.02, 0.08, 0.06, 0.05]
STDS_OCOC = [0.05, 0.04, 0.02, 0.03, 0.07, 0.06]
STDS_SGSG = [0.06, 0.05, 0.03, 0.04, 0.08, 0.07]


def configure_style() -> None:
    available = {f.name for f in fm.fontManager.ttflist}
    preferred = ["Arial", "Helvetica", "Liberation Sans"]
    chosen = next((name for name in preferred if name in available), None)
    if chosen is None:
        raise RuntimeError("None of Arial, Helvetica, or Liberation Sans is available.")

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [chosen],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "axes.titleweight": "normal",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 9,
            "legend.frameon": False,
            "lines.linewidth": 1.5,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.format": "pdf",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "hatch.linewidth": 0.8,
        }
    )


def main() -> None:
    configure_style()

    fig, ax = plt.subplots(figsize=(9, 4.5))

    n_classes = len(CLASSES)
    bar_width = 0.27
    x = np.arange(n_classes)

    bars_bb = ax.bar(
        x - bar_width,
        MEANS_BB,
        bar_width,
        yerr=STDS_BB,
        label="B-B (OpenAI CLIP)",
        color=PALETTE["b_b"],
        edgecolor="white",
        linewidth=0.5,
        capsize=3,
        ecolor="black",
        zorder=3,
    )
    bars_oc = ax.bar(
        x,
        MEANS_OCOC,
        bar_width,
        yerr=STDS_OCOC,
        label="OC-OC (LAION-2B)",
        color=PALETTE["oc_oc"],
        edgecolor="white",
        linewidth=0.5,
        capsize=3,
        ecolor="black",
        zorder=3,
    )
    bars_sg = ax.bar(
        x + bar_width,
        MEANS_SGSG,
        bar_width,
        yerr=STDS_SGSG,
        label="SG-SG (SigLIP)",
        color=PALETTE["sg_sg"],
        edgecolor="white",
        linewidth=0.5,
        capsize=3,
        ecolor="black",
        zorder=3,
    )

    highlights: list[tuple[str, str, str]] = []
    for i in range(n_classes):
        bb = MEANS_BB[i]
        oc = MEANS_OCOC[i]
        sg = MEANS_SGSG[i]

        if bb <= 0.5:
            bars_bb[i].set_edgecolor(PALETTE["cliff"])
            bars_bb[i].set_linewidth(2.5)
            bars_bb[i].set_hatch("//")
            highlights.append((CLASSES[i].replace("\n", " "), "B-B", "cliff"))

            if oc > 0.7:
                bars_oc[i].set_edgecolor(PALETTE["closure"])
                bars_oc[i].set_linewidth(2.5)
                highlights.append((CLASSES[i].replace("\n", " "), "OC-OC", "closure"))
            if sg > 0.7:
                bars_sg[i].set_edgecolor(PALETTE["closure"])
                bars_sg[i].set_linewidth(2.5)
                highlights.append((CLASSES[i].replace("\n", " "), "SG-SG", "closure"))

    ax.axhline(
        0.5,
        color=PALETTE["chance"],
        linestyle="--",
        alpha=0.4,
        linewidth=1.0,
        zorder=1,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=9)
    ax.set_ylabel("3-seed mean accuracy", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    assert len(bars_bb) == 6, f"Expected 6 B-B bars, got {len(bars_bb)}"
    assert all(b.get_height() > 0 for b in bars_bb), "Some B-B bars have 0 height"
    assert all(b.get_height() > 0 for b in bars_oc), "Some OC-OC bars have 0 height"
    assert all(b.get_height() > 0 for b in bars_sg), "Some SG-SG bars have 0 height"

    out_path = Path("fig_cross_family_v2.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("Bars drawn (heights):")
    print("  B-B   :", [round(float(b.get_height()), 3) for b in bars_bb])
    print("  OC-OC :", [round(float(b.get_height()), 3) for b in bars_oc])
    print("  SG-SG :", [round(float(b.get_height()), 3) for b in bars_sg])
    print("Highlights:")
    for cls, system, kind in highlights:
        print(f"  {cls}\t{system}\t{kind}")
    print(f"saved\t{out_path}")


if __name__ == "__main__":
    main()
