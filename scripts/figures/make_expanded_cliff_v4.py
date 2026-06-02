"""Figure 3 — figure_expanded_cliff_B-B_v4.

Upgrade of figures_new/figure_expanded_cliff_B-B_v3 with 3 polish fixes:
  1. Motion LOWESS — frac=0.65, it=3, evaluated on a dense x-grid via
     statsmodels lowess(xvals=...) for a smoother curve that spans the
     full data range.
  2. Per-panel adaptive annotation placement (avoids data overlap):
       y_max <= 0.4              → upper-right (0.97, 0.95) va=top
       n_fail >= 2 AND n_pass >= 2 → mid-right  (0.97, 0.55) va=center
       otherwise                  → lower-right (0.97, 0.05) va=bottom
  3. Smaller markers: pass 50, fail 60, white edge 0.6, alpha 0.9.

Identical to v3:
  - Source JSONs (experiments/EXP-A/eval/B-B/seed{1,2,3})
  - Dict-insertion-order stats arrays (bootstrap RNG fidelity)
  - Pearson r + motion 95% CI validation (< 0.001 tolerance)
  - 5-panel horizontal layout, panel order, axis labels

Output: figures_camera_ready/figure_expanded_cliff_B-B_v4.pdf + .png
"""
from __future__ import annotations
import json
import glob
import sys
import subprocess
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
from statsmodels.nonparametric.smoothers_lowess import lowess

ROOT = Path("/path/to/project")
sys.path.insert(0, str(ROOT / ".skills/nature-figure-style"))
from style import apply_nature_style, PALETTE
from helpers import add_chance_line

OUT_DIR = ROOT / "figures_camera_ready"
OUT_PDF = OUT_DIR / "figure_expanded_cliff_B-B_v4.pdf"
OUT_PNG = OUT_DIR / "figure_expanded_cliff_B-B_v4.png"

PANEL_ORDER = ["motion", "size_BIG", "size_SMALL", "speed_FAST", "speed_SLOW"]
DISPLAY = {
    "motion":      "motion",
    "size_BIG":    "size BIG",
    "size_SMALL":  "size SMALL",
    "speed_FAST":  "speed FAST",
    "speed_SLOW":  "speed SLOW",
}
EXPECTED_R = {
    "motion":      0.627,
    "size_BIG":   -0.488,
    "size_SMALL":  0.434,
    "speed_FAST":  0.760,
    "speed_SLOW": -0.508,
}
EXPECTED_MOTION_CI = (0.158, 0.916)


def load_per_token():
    """Aggregate 3-seed JSONs preserving dict-insertion order for RNG fidelity."""
    paths = sorted(glob.glob(str(ROOT / "experiments/EXP-A/eval/B-B/seed*/expanded_eval_seed*.json")))
    if not paths:
        raise FileNotFoundError("no expanded_eval JSONs")
    per_token_seeds: dict[str, list[float]] = {}
    feat: dict[str, dict] = {}
    for p in paths:
        d = json.load(open(p))
        for tok, info in d["per_token"].items():
            if info.get("accuracy") is None:
                continue
            per_token_seeds.setdefault(tok, []).append(info["accuracy"])
            if tok not in feat:
                feat[tok] = {"axis_class": info["axis_class"],
                             "cos_b32": info["cos_b32"]}
    out = {}
    for tok, accs in per_token_seeds.items():
        out[tok] = dict(feat[tok])
        out[tok]["acc_mean"] = float(np.mean(accs))
    return out


def pearson_bootstrap(x, y, n_boot=1000, seed=0):
    x = np.asarray(x, float); y = np.asarray(y, float)
    r = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        if np.std(xb) < 1e-9 or np.std(yb) < 1e-9:
            continue
        boots.append(np.corrcoef(xb, yb)[0, 1])
    boots = np.array(boots)
    return r, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def annotation_anchor(acc_array):
    """Adaptive (x, y, va) for the r-annotation in this panel."""
    y_max = float(np.max(acc_array))
    n_fail = int(np.sum(acc_array <= 0.5))
    n_pass = int(np.sum(acc_array > 0.5))
    if y_max <= 0.4:
        return 0.97, 0.95, "top"
    if n_fail >= 2 and n_pass >= 2:
        return 0.97, 0.55, "center"
    return 0.97, 0.05, "bottom"


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
    per_token = load_per_token()

    # Stats arrays preserve dict insertion order (RNG fidelity);
    # plot arrays separately sorted by cos for monotone scatter.
    stats_pts: dict[str, list[dict]] = {c: [] for c in PANEL_ORDER}
    for tok, info in per_token.items():
        c = info["axis_class"]
        if c in stats_pts:
            stats_pts[c].append({"token": tok, "cos": info["cos_b32"],
                                 "acc": info["acc_mean"]})
    plot_pts = {c: sorted(stats_pts[c], key=lambda d: d["cos"])
                for c in PANEL_ORDER}

    # Pearson r per panel + CI for motion (UNSORTED arrays)
    rs: dict[str, float] = {}
    motion_ci = None
    for c in PANEL_ORDER:
        pts = stats_pts[c]
        cos = [p["cos"] for p in pts]; acc = [p["acc"] for p in pts]
        if c == "motion":
            r, lo, hi = pearson_bootstrap(cos, acc, n_boot=1000, seed=0)
            rs[c] = r; motion_ci = (lo, hi)
        else:
            rs[c] = float(np.corrcoef(cos, acc)[0, 1])

    # Validate
    print("=" * 78)
    print("VALIDATION (tolerance < 0.001)")
    print("=" * 78)
    bad = []
    for c in PANEL_ORDER:
        diff = rs[c] - EXPECTED_R[c]
        ok = abs(diff) < 0.001
        print(f"  {DISPLAY[c]:12s} r={rs[c]:+.6f} exp={EXPECTED_R[c]:+.4f} Δ={diff:+.6f} [{'OK' if ok else 'BAD'}]")
        if not ok: bad.append(c)
    d_lo = motion_ci[0] - EXPECTED_MOTION_CI[0]
    d_hi = motion_ci[1] - EXPECTED_MOTION_CI[1]
    ok_ci = abs(d_lo) < 0.001 and abs(d_hi) < 0.001
    print(f"  motion CI [{motion_ci[0]:+.4f}, {motion_ci[1]:+.4f}]"
          f" exp [{EXPECTED_MOTION_CI[0]:+.4f}, {EXPECTED_MOTION_CI[1]:+.4f}]"
          f" Δ=[{d_lo:+.4f}, {d_hi:+.4f}] [{'OK' if ok_ci else 'BAD'}]")
    if bad or not ok_ci:
        raise RuntimeError(f"validation failed")

    # Plot
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    for ax_idx, c in enumerate(PANEL_ORDER):
        ax = axes[ax_idx]
        pts = plot_pts[c]
        cos = np.array([p["cos"] for p in pts])
        acc = np.array([p["acc"] for p in pts])
        fail = acc <= 0.5

        # Chance line (label on first panel only)
        add_chance_line(ax, y=0.5,
                        label=("chance" if ax_idx == 0 else None),
                        label_pos="left")

        # Motion-only LOWESS (frac=0.65, it=3, dense x-grid)
        if c == "motion" and len(cos) >= 4:
            x_grid = np.linspace(cos.min(), cos.max(), 100)
            smoothed = lowess(acc, cos, frac=0.65, it=3,
                              xvals=x_grid, return_sorted=False)
            ax.plot(x_grid, smoothed, color=PALETTE["neutral"],
                    linewidth=1.8, alpha=0.7, zorder=2)

        # Scatter — pass 50, fail 60
        if fail.any():
            ax.scatter(cos[fail], acc[fail], s=60,
                       c=PALETTE["cliff"], marker="D",
                       edgecolors="white", linewidths=0.6, alpha=0.9, zorder=3)
        if (~fail).any():
            ax.scatter(cos[~fail], acc[~fail], s=50,
                       c=PALETTE["neutral"], marker="o",
                       edgecolors="white", linewidths=0.6, alpha=0.9, zorder=3)

        # Adaptive annotation
        x_anchor, y_anchor, va = annotation_anchor(acc)
        if c == "motion":
            text = f"r={rs[c]:.3f}, CI[{motion_ci[0]:.3f}, {motion_ci[1]:.3f}]"
        else:
            text = f"r={rs[c]:.3f}"
        ax.text(x_anchor, y_anchor, text, transform=ax.transAxes,
                ha="right", va=va, fontsize=9, color="black",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="none", alpha=0.85))

        ax.set_title(f"{DISPLAY[c]} (n={len(pts)})", pad=6)
        ax.set_xlabel("CLIP-B/32 cosine")
        ax.set_ylim(0.0, 1.0)
    axes[0].set_ylabel("3-seed mean accuracy")

    fig.savefig(OUT_PDF, bbox_inches="tight", dpi=300)
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=300)
    plt.close(fig)

    # Report annotation placement (sanity)
    print("\n## Adaptive annotation placement per panel")
    print(f"  {'panel':12s} {'y_max':>7s} {'n_fail':>7s} {'n_pass':>7s} {'placement':>14s}")
    for c in PANEL_ORDER:
        pts = stats_pts[c]
        acc = np.array([p["acc"] for p in pts])
        y_max = float(acc.max())
        n_fail = int((acc <= 0.5).sum())
        n_pass = int((acc > 0.5).sum())
        x_a, y_a, va = annotation_anchor(acc)
        placement = f"({x_a:.2f},{y_a:.2f}) va={va}"
        print(f"  {DISPLAY[c]:12s} {y_max:>7.3f} {n_fail:>7d} {n_pass:>7d} {placement:>22s}")

    print()
    verify_render(OUT_PDF, OUT_PNG)
    print()
    print(f"PDF: {OUT_PDF.resolve()}")
    print(f"PNG: {OUT_PNG.resolve()}")


if __name__ == "__main__":
    main()
