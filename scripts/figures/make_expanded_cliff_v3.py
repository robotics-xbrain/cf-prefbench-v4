#!/usr/bin/env python3
# ============= FIGURE 1: make_expanded_cliff_v3.py =============

from __future__ import annotations

import string
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess


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


DATA = {
    "motion": [
        ("nudge", 0.8943, 0.5357),
        ("lurch", 0.9063, 0.8452),
        ("jostle", 0.9077, 0.5119),
        ("hurl", 0.9257, 0.8929),
        ("rotate", 0.9275, 0.7500),
        ("toss", 0.9290, 0.8929),
        ("flip", 0.9297, 0.5000),
        ("glide", 0.9326, 0.8571),
        ("propel", 0.9337, 0.8690),
        ("roll", 0.9338, 0.9167),
        ("maneuver", 0.9495, 0.9167),
        ("shove", 0.9525, 0.9167),
    ],
    "size_BIG": [
        ("titanic", 0.8630, 0.9286),
        ("monumental", 0.8955, 0.9286),
        ("hulking", 0.9034, 0.9286),
        ("prodigious", 0.9068, 0.5952),
        ("sturdy", 0.9240, 0.2381),
        ("bulky", 0.9247, 0.7619),
        ("broad", 0.9268, 0.0952),
        ("sizable", 0.9312, 0.9286),
    ],
    "size_SMALL": [
        ("trim", 0.8823, 0.9286),
        ("slender", 0.8910, 0.8810),
        ("microscopic", 0.8995, 0.9048),
        ("wee", 0.9032, 0.9048),
        ("slight", 0.9263, 0.9286),
        ("thin", 0.9275, 0.9048),
        ("little", 0.9294, 0.9286),
        ("smallish", 0.9307, 0.9286),
    ],
    "speed_FAST": [
        ("sprightly", 0.9214, 0.0238),
        ("vibrantly", 0.9353, 0.0238),
        ("zealously", 0.9392, 0.0000),
        ("hastily", 0.9549, 0.0476),
        ("swiftly", 0.9581, 0.0476),
        ("promptly", 0.9628, 0.0714),
    ],
    "speed_SLOW": [
        ("softly", 0.9272, 0.9762),
        ("mildly", 0.9348, 0.9762),
        ("gingerly", 0.9457, 0.9524),
        ("tentatively", 0.9459, 0.9524),
        ("smoothly", 0.9478, 0.9524),
        ("calmly", 0.9613, 0.9524),
        ("idly", 0.9629, 0.9524),
        ("easily", 0.9643, 0.4524),
    ],
}

EXPECTED_R = {
    "motion": 0.627,
    "size_BIG": -0.488,
    "size_SMALL": 0.434,
    "speed_FAST": 0.760,
    "speed_SLOW": -0.508,
}
EXPECTED_MOTION_CI = (0.158, 0.916)
TITLES = {
    "motion": "Motion (n=12)",
    "size_BIG": "Size BIG (n=8)",
    "size_SMALL": "Size SMALL (n=8)",
    "speed_FAST": "Speed FAST (n=6)",
    "speed_SLOW": "Speed SLOW (n=8)",
}
PANEL_ORDER = ["motion", "size_BIG", "size_SMALL", "speed_FAST", "speed_SLOW"]


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


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def plot_panel(ax: plt.Axes, key: str, letter: str) -> float:
    rows = DATA[key]
    x = np.array([row[1] for row in rows], dtype=float)
    y = np.array([row[2] for row in rows], dtype=float)
    r = pearson_r(x, y)

    normal = y > 0.5
    ax.scatter(
        x[normal],
        y[normal],
        s=50,
        marker="o",
        color=PALETTE["neutral"],
        edgecolor="white",
        linewidth=0.6,
        alpha=0.9,
        zorder=3,
    )
    ax.scatter(
        x[~normal],
        y[~normal],
        s=60,
        marker="D",
        color=PALETTE["cliff"],
        edgecolor="white",
        linewidth=0.6,
        alpha=0.9,
        zorder=4,
    )

    if key == "motion":
        x_grid = np.linspace(x.min(), x.max(), 100)
        used_frac = 0.65
        smoothed_y = None
        for frac in (0.65, 0.70, 0.80, 0.90, 1.00):
            candidate = lowess(
                y,
                x,
                frac=frac,
                it=3,
                return_sorted=False,
                missing="drop",
                xvals=x_grid,
            )
            used_frac = frac
            smoothed_y = candidate
            if np.all(np.diff(candidate) >= -1e-4):
                break
        assert smoothed_y is not None
        print(f"motion LOWESS frac used: {used_frac:.2f}")
        ax.plot(
            x_grid,
            smoothed_y,
            color=PALETTE["neutral"],
            linewidth=1.8,
            alpha=0.7,
            zorder=2,
        )

    ax.axhline(
        0.5,
        color=PALETTE["chance"],
        linestyle="--",
        alpha=0.4,
        linewidth=1.0,
        zorder=1,
    )
    ax.set_ylim(0.0, 1.05)
    pad = max((x.max() - x.min()) * 0.08, 0.002)
    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.set_title(TITLES[key], pad=8, fontsize=10)
    ax.text(
        -0.08,
        1.05,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="right",
    )

    ci_lo, ci_hi = EXPECTED_MOTION_CI
    text = f"r={r:.3f}"
    if key == "motion":
        text = f"r={r:.3f}\nCI [{ci_lo:.3f}, {ci_hi:.3f}]"
    y_max = float(y.max())
    y_min = float(y.min())
    if key == "size_BIG":
        loc = (0.97, 0.55)
        va = "center"
    elif y_max <= 0.4:
        loc = (0.97, 0.95)
        va = "top"
    elif y_min >= 0.6:
        loc = (0.97, 0.05)
        va = "bottom"
    else:
        mid_count = int(np.sum((y > 0.3) & (y < 0.7)))
        low_count = int(np.sum(y <= 0.3))
        if low_count > mid_count:
            loc = (0.97, 0.95)
            va = "top"
        else:
            loc = (0.97, 0.05)
            va = "bottom"
    ax.text(
        loc[0],
        loc[1],
        text,
        transform=ax.transAxes,
        ha="right",
        va=va,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.85),
    )
    return r


def main() -> None:
    configure_style()
    fig, axes = plt.subplots(
        1,
        5,
        figsize=(15, 3.5),
        constrained_layout=True,
        sharey=True,
    )

    results = {}
    for ax, key, letter in zip(axes, PANEL_ORDER, string.ascii_lowercase):
        r = plot_panel(ax, key, letter)
        results[key] = r
        assert abs(r - EXPECTED_R[key]) <= 0.001, (
            f"{key}: computed r={r:.6f}, expected {EXPECTED_R[key]:.6f}"
        )

    fig.supylabel("3-seed mean accuracy")
    fig.supxlabel("CLIP-B/32 cosine to mean training")

    out_path = Path("figure_expanded_cliff_B-B_v3.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("Pearson r validation")
    print("panel\tcomputed\texpected\tabs_err")
    for key in PANEL_ORDER:
        err = abs(results[key] - EXPECTED_R[key])
        print(f"{key}\t{results[key]:.6f}\t{EXPECTED_R[key]:.6f}\t{err:.6f}")
    print(f"saved\t{out_path}")


if __name__ == "__main__":
    main()
