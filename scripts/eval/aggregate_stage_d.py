"""Stage D: aggregate per-token & per-class accuracies over the expanded pool.

Combines:
  - 8 existing cliff tokens (colossal/gigantic/miniature/petite/briskly/speedily/sluggishly/gradually)
    from EXP-B/B-B_published + EXP-F/{B-32,L-14}/seed* JSONs (already on disk).
  - 8 new tokens (enormous/vast/tiny/minute/swiftly/hastily/leisurely/languidly) from
    realdata_validation/expanded_tokens/predictions/*.json produced by Stage C.

Computes per-token n=8-seed mean + 95% bootstrap CI (1000 resamples) and per-class means
over the combined token pool (n=4 tokens × 8 seeds = 32 datapoints per cell).

Outputs:
  - realdata_validation/expanded_tokens/tables/table_size_per_token.csv   (8 size tokens × backbones)
  - realdata_validation/expanded_tokens/tables/table_speed_per_token.csv  (8 speed tokens × backbones)
  - realdata_validation/expanded_tokens/tables/table_3axis_n16.csv        (class means: old8 vs combined16 vs new8)
  - realdata_validation/expanded_tokens/tables/per_seed_per_token.json    (raw lookup)
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/path/to/project")
PRED_DIR = ROOT / "realdata_validation/expanded_tokens/predictions"
TABLES = ROOT / "realdata_validation/expanded_tokens/tables"
TABLES.mkdir(parents=True, exist_ok=True)

EXISTING_TOKENS = {
    "size":  ["colossal", "gigantic", "miniature", "petite"],
    "speed": ["briskly", "speedily", "sluggishly", "gradually"],
}
NEW_TOKENS = {
    "size":  ["enormous", "vast", "tiny", "minute"],
    "speed": ["swiftly", "hastily", "leisurely", "languidly"],
}
SEM_CLASS = {
    "colossal": "BIG", "gigantic": "BIG",
    "miniature": "SMALL", "petite": "SMALL",
    "briskly": "FAST", "speedily": "FAST",
    "sluggishly": "SLOW", "gradually": "SLOW",
    "enormous": "BIG", "vast": "BIG",
    "tiny": "SMALL", "minute": "SMALL",
    "swiftly": "FAST", "hastily": "FAST",
    "leisurely": "SLOW", "languidly": "SLOW",
}
SEEDS = [1, 2, 3, 7, 31, 99, 256, 2025]


def load_existing_per_seed_per_token(backbone: str) -> dict:
    """backbone ∈ {'b32','l14'}.

    Returns {token: {seed: {accuracy, cos_to_train, n_rows}}}.
    Sourced from:
      - published seeds 1/2/3: experiments/EXP-B/{B-B_published,L-L_published}/seed{S}/cliff_tokens_seed*.json
      - EXP-F seeds 7/31/99/256/2025: experiments/EXP-F/{B-32,L-14}/seed{S}/cliff_tokens_seed*.json
    """
    pub_dir = "B-B_published" if backbone == "b32" else "L-L_published"
    new_dir = "B-32" if backbone == "b32" else "L-14"
    paths: list[tuple[int, Path]] = []
    for s in [1, 2, 3]:
        p = ROOT / f"experiments/EXP-B/{pub_dir}/seed{s}/cliff_tokens_seed{s}.json"
        if p.exists():
            paths.append((s, p))
    for s in [7, 31, 99, 256, 2025]:
        p = ROOT / f"experiments/EXP-F/{new_dir}/seed{s}/cliff_tokens_seed{s}.json"
        if p.exists():
            paths.append((s, p))
    out: dict = defaultdict(dict)
    for seed, p in paths:
        d = json.loads(p.read_text())
        for tok in EXISTING_TOKENS["size"] + EXISTING_TOKENS["speed"]:
            info = d["per_token"].get(tok)
            if info is None:
                continue
            out[tok][seed] = {
                "accuracy": info["accuracy"],
                "cos_to_train": info["cos_to_train"],
                "n_rows": info["n_rows"],
            }
    return out


def load_new_per_seed_per_token(backbone: str) -> dict:
    """Same shape, but for the 8 new tokens from Stage C predictions."""
    out: dict = defaultdict(dict)
    for s in SEEDS:
        p = PRED_DIR / f"{backbone}_seed{s}_new_tokens.json"
        if not p.exists():
            print(f"  WARN missing prediction: {p}")
            continue
        d = json.loads(p.read_text())
        for tok, info in d["per_token"].items():
            out[tok][s] = {
                "accuracy": info["accuracy"],
                "cos_to_train": info["cos_to_train"],
                "n_rows": info["n_rows"],
            }
    return out


def bootstrap_ci(vals: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Return (mean, lo, hi) using percentile bootstrap on `vals` (1-D)."""
    rng = np.random.default_rng(seed)
    n = len(vals)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = vals[idx].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (float(vals.mean()), lo, hi)


def paired_bootstrap_ci(per_seed_per_token: dict, tokens: list[str],
                        seeds: list[int], n_boot: int = 1000,
                        alpha: float = 0.05, seed: int = 0):
    """Paired-by-seed bootstrap of the class mean.

    Build the per-seed token-averaged accuracy y_s = mean over tokens of acc(s, token).
    Resample seeds with replacement; the bootstrap statistic is mean(y over resampled seeds).
    This accounts for correlation: a "lucky" seed is lucky on all its tokens together.

    Returns (mean, lo, hi, n_seeds_used).
    """
    # For each seed, compute mean accuracy across the tokens that have a value for that seed.
    per_seed_means: list[float] = []
    for s in seeds:
        accs = [per_seed_per_token.get(tok, {}).get(s, {}).get("accuracy") for tok in tokens]
        accs = [a for a in accs if a is not None]
        if not accs:
            continue
        per_seed_means.append(float(np.mean(accs)))
    y = np.array(per_seed_means, dtype=np.float64)
    n = len(y)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = y[idx].mean()
    return (float(y.mean()), float(np.quantile(boot, alpha / 2)),
            float(np.quantile(boot, 1 - alpha / 2)), n)


def per_token_summary(per_seed: dict, token: str) -> dict:
    rec = per_seed.get(token, {})
    accs = np.array([v["accuracy"] for v in rec.values()], dtype=np.float64)
    coses = np.array([v["cos_to_train"] for v in rec.values()], dtype=np.float64)
    m, lo, hi = bootstrap_ci(accs)
    return {
        "n_seeds": len(accs),
        "mean": m, "ci95_lo": lo, "ci95_hi": hi,
        "std": float(accs.std(ddof=0)) if len(accs) else float("nan"),
        "cos_to_train_mean": float(coses.mean()) if len(coses) else float("nan"),
    }


def class_pool(per_seed: dict, tokens: list[str]) -> np.ndarray:
    """Pool all (token, seed) accuracies for the class — shape (n_tokens * n_seeds,)."""
    vals: list[float] = []
    for tok in tokens:
        for s in SEEDS:
            v = per_seed.get(tok, {}).get(s)
            if v is not None:
                vals.append(v["accuracy"])
    return np.array(vals, dtype=np.float64)


def write_per_token_csv(axis: str, b32_existing: dict, b32_new: dict,
                       l14_existing: dict, l14_new: dict) -> None:
    """Write tables/table_{axis}_per_token.csv with 8 rows (4 existing + 4 new)."""
    out_path = TABLES / f"table_{axis}_per_token.csv"
    tokens = EXISTING_TOKENS[axis] + NEW_TOKENS[axis]
    is_new = {t: (t in NEW_TOKENS[axis]) for t in tokens}

    with out_path.open("w") as fh:
        w = csv.writer(fh)
        w.writerow(["axis", "sem_class", "token", "source",
                    "b32_n_seeds", "b32_mean", "b32_ci95_lo", "b32_ci95_hi", "b32_std",
                    "b32_cos_to_train",
                    "l14_n_seeds", "l14_mean", "l14_ci95_lo", "l14_ci95_hi", "l14_std",
                    "l14_cos_to_train"])
        for tok in tokens:
            b32_src = b32_new if is_new[tok] else b32_existing
            l14_src = l14_new if is_new[tok] else l14_existing
            b = per_token_summary(b32_src, tok)
            l = per_token_summary(l14_src, tok)
            w.writerow([axis, SEM_CLASS[tok], tok, "new" if is_new[tok] else "paper",
                        b["n_seeds"], f"{b['mean']:.4f}", f"{b['ci95_lo']:.4f}",
                        f"{b['ci95_hi']:.4f}", f"{b['std']:.4f}",
                        f"{b['cos_to_train_mean']:.4f}",
                        l["n_seeds"], f"{l['mean']:.4f}", f"{l['ci95_lo']:.4f}",
                        f"{l['ci95_hi']:.4f}", f"{l['std']:.4f}",
                        f"{l['cos_to_train_mean']:.4f}"])
    print(f"  wrote {out_path}")


def write_3axis_csv(b32_existing: dict, b32_new: dict,
                    l14_existing: dict, l14_new: dict) -> None:
    """Write n=16 class-level table with BOTH unpaired (token×seed pool) and
    paired-by-seed bootstrap CIs."""
    out_path = TABLES / "table_3axis_n16.csv"
    paired_out = TABLES / "table_3axis_n16_paired.csv"
    class_defs = [
        ("size",  "BIG",   "enormous,vast",         "colossal,gigantic"),
        ("size",  "SMALL", "tiny,minute",           "miniature,petite"),
        ("speed", "FAST",  "swiftly,hastily",       "briskly,speedily"),
        ("speed", "SLOW",  "leisurely,languidly",   "sluggishly,gradually"),
    ]

    # ---- unpaired (token×seed pool) bootstrap (original method) ----
    with out_path.open("w") as fh:
        w = csv.writer(fh)
        w.writerow(["axis", "class",
                    "n_tokens_old", "n_tokens_new", "n_tokens_combined",
                    "n_seeds",
                    "b32_old_mean", "b32_old_ci95_lo", "b32_old_ci95_hi", "b32_old_n",
                    "b32_new_mean", "b32_new_ci95_lo", "b32_new_ci95_hi", "b32_new_n",
                    "b32_combined_mean", "b32_combined_ci95_lo", "b32_combined_ci95_hi", "b32_combined_n",
                    "l14_old_mean", "l14_old_ci95_lo", "l14_old_ci95_hi", "l14_old_n",
                    "l14_new_mean", "l14_new_ci95_lo", "l14_new_ci95_hi", "l14_new_n",
                    "l14_combined_mean", "l14_combined_ci95_lo", "l14_combined_ci95_hi", "l14_combined_n"])
        for axis, cls, new_toks, old_toks in class_defs:
            old_list = old_toks.split(",")
            new_list = new_toks.split(",")
            b32_old = class_pool(b32_existing, old_list)
            b32_new_pool = class_pool(b32_new, new_list)
            b32_comb = np.concatenate([b32_old, b32_new_pool])
            l14_old = class_pool(l14_existing, old_list)
            l14_new_pool = class_pool(l14_new, new_list)
            l14_comb = np.concatenate([l14_old, l14_new_pool])
            bm = lambda v: bootstrap_ci(v)
            mo, lo_o, ho = bm(b32_old)
            mn, lo_n, hn = bm(b32_new_pool)
            mc, lo_c, hc = bm(b32_comb)
            lmo, llo_o, lho = bm(l14_old)
            lmn, llo_n, lhn = bm(l14_new_pool)
            lmc, llo_c, lhc = bm(l14_comb)
            w.writerow([axis, cls, 2, 2, 4, 8,
                        f"{mo:.4f}", f"{lo_o:.4f}", f"{ho:.4f}", len(b32_old),
                        f"{mn:.4f}", f"{lo_n:.4f}", f"{hn:.4f}", len(b32_new_pool),
                        f"{mc:.4f}", f"{lo_c:.4f}", f"{hc:.4f}", len(b32_comb),
                        f"{lmo:.4f}", f"{llo_o:.4f}", f"{lho:.4f}", len(l14_old),
                        f"{lmn:.4f}", f"{llo_n:.4f}", f"{lhn:.4f}", len(l14_new_pool),
                        f"{lmc:.4f}", f"{llo_c:.4f}", f"{lhc:.4f}", len(l14_comb)])
    print(f"  wrote {out_path}")

    # ---- paired-by-seed bootstrap (treats seeds as the resampling unit) ----
    with paired_out.open("w") as fh:
        w = csv.writer(fh)
        w.writerow(["axis", "class",
                    "n_tokens_old", "n_tokens_new", "n_tokens_combined",
                    "b32_old_mean", "b32_old_ci95_lo_paired", "b32_old_ci95_hi_paired",
                    "b32_new_mean", "b32_new_ci95_lo_paired", "b32_new_ci95_hi_paired",
                    "b32_combined_mean", "b32_combined_ci95_lo_paired", "b32_combined_ci95_hi_paired",
                    "l14_old_mean", "l14_old_ci95_lo_paired", "l14_old_ci95_hi_paired",
                    "l14_new_mean", "l14_new_ci95_lo_paired", "l14_new_ci95_hi_paired",
                    "l14_combined_mean", "l14_combined_ci95_lo_paired", "l14_combined_ci95_hi_paired"])
        for axis, cls, new_toks, old_toks in class_defs:
            old_list = old_toks.split(",")
            new_list = new_toks.split(",")
            comb_list = old_list + new_list

            # B/32
            b_o_mean, b_o_lo, b_o_hi, _ = paired_bootstrap_ci(b32_existing, old_list, SEEDS)
            b_n_mean, b_n_lo, b_n_hi, _ = paired_bootstrap_ci(b32_new, new_list, SEEDS)
            # Combined needs a merged seed-aligned lookup
            merged_b32 = {**b32_existing, **b32_new}
            b_c_mean, b_c_lo, b_c_hi, _ = paired_bootstrap_ci(merged_b32, comb_list, SEEDS)
            # L/14
            l_o_mean, l_o_lo, l_o_hi, _ = paired_bootstrap_ci(l14_existing, old_list, SEEDS)
            l_n_mean, l_n_lo, l_n_hi, _ = paired_bootstrap_ci(l14_new, new_list, SEEDS)
            merged_l14 = {**l14_existing, **l14_new}
            l_c_mean, l_c_lo, l_c_hi, _ = paired_bootstrap_ci(merged_l14, comb_list, SEEDS)

            w.writerow([axis, cls, 2, 2, 4,
                        f"{b_o_mean:.4f}", f"{b_o_lo:.4f}", f"{b_o_hi:.4f}",
                        f"{b_n_mean:.4f}", f"{b_n_lo:.4f}", f"{b_n_hi:.4f}",
                        f"{b_c_mean:.4f}", f"{b_c_lo:.4f}", f"{b_c_hi:.4f}",
                        f"{l_o_mean:.4f}", f"{l_o_lo:.4f}", f"{l_o_hi:.4f}",
                        f"{l_n_mean:.4f}", f"{l_n_lo:.4f}", f"{l_n_hi:.4f}",
                        f"{l_c_mean:.4f}", f"{l_c_lo:.4f}", f"{l_c_hi:.4f}"])
    print(f"  wrote {paired_out}")


def detect_surprises(b32_existing: dict, b32_new: dict,
                     l14_existing: dict, l14_new: dict) -> dict:
    """Score each token against three thresholds:
      - cliff_fail:    B/32 mean <= 0.50  (cliff-confirming behavior)
      - ceiling_pass:  B/32 mean >= 0.75  (near-ceiling behavior)
      - borderline:    0.50 < mean < 0.75 (neither clear cliff nor clear pass)

    A "surprise" is any token whose B/32 behavior contradicts its cliff-class
    prediction (BIG/SLOW expected ceiling_pass; SMALL/FAST expected cliff_fail).
    We additionally report L/14 to assess whether scaling resolves the surprise.
    """
    CLIFF_THR = 0.50
    CEIL_THR = 0.75
    expected_by_class = {"BIG": "ceiling_pass", "SMALL": "cliff_fail",
                         "FAST": "cliff_fail", "SLOW": "ceiling_pass"}

    def classify(m: float) -> str:
        if m <= CLIFF_THR: return "cliff_fail"
        if m >= CEIL_THR:  return "ceiling_pass"
        return "borderline"

    by_class = {
        "BIG":   {"paper": ["colossal", "gigantic"], "new": ["enormous", "vast"]},
        "SMALL": {"paper": ["miniature", "petite"],  "new": ["tiny", "minute"]},
        "FAST":  {"paper": ["briskly", "speedily"],  "new": ["swiftly", "hastily"]},
        "SLOW":  {"paper": ["sluggishly", "gradually"], "new": ["leisurely", "languidly"]},
    }
    per_token_rows: list[dict] = []
    surprises: list[dict] = []

    for cls, src_map in by_class.items():
        expected = expected_by_class[cls]
        for source, toks in src_map.items():
            b_src = b32_existing if source == "paper" else b32_new
            l_src = l14_existing if source == "paper" else l14_new
            for tok in toks:
                b_accs = [v["accuracy"] for v in b_src.get(tok, {}).values()]
                l_accs = [v["accuracy"] for v in l_src.get(tok, {}).values()]
                if not b_accs:
                    continue
                b_mean = float(np.mean(b_accs))
                l_mean = float(np.mean(l_accs)) if l_accs else float("nan")
                b_outcome = classify(b_mean)
                l_outcome = classify(l_mean) if not np.isnan(l_mean) else None
                rec = {
                    "source": source, "class": cls, "token": tok,
                    "b32_mean": round(b_mean, 4), "l14_mean": round(l_mean, 4),
                    "b32_outcome": b_outcome, "l14_outcome": l_outcome,
                    "expected": expected,
                    "agrees_expected": (b_outcome == expected),
                }
                per_token_rows.append(rec)
                if b_outcome != expected:
                    direction = (f"expected={expected}, got={b_outcome}")
                    surprises.append({**rec, "direction": direction})

    # Summary counts
    counts = defaultdict(lambda: {"cliff_fail": 0, "borderline": 0, "ceiling_pass": 0,
                                  "n_total": 0, "n_agree": 0, "n_disagree": 0})
    for r in per_token_rows:
        cls = r["class"]; out = r["b32_outcome"]
        counts[cls][out] += 1
        counts[cls]["n_total"] += 1
        if r["agrees_expected"]:
            counts[cls]["n_agree"] += 1
        else:
            counts[cls]["n_disagree"] += 1

    return {
        "rule": ("Per-token B/32 outcome classified by accuracy: "
                 "cliff_fail (mean ≤ 0.50), ceiling_pass (mean ≥ 0.75), borderline otherwise. "
                 "Expected outcome by class: BIG/SLOW ceiling_pass, SMALL/FAST cliff_fail."),
        "thresholds": {"cliff_fail_max": CLIFF_THR, "ceiling_pass_min": CEIL_THR},
        "per_token": per_token_rows,
        "class_counts": dict(counts),
        "surprises": surprises,
    }


def main() -> None:
    print("[stage-d] loading existing per-token results…")
    b32_existing = load_existing_per_seed_per_token("b32")
    l14_existing = load_existing_per_seed_per_token("l14")
    print(f"  b32 existing: {sum(len(v) for v in b32_existing.values())} (token,seed) datapoints")
    print(f"  l14 existing: {sum(len(v) for v in l14_existing.values())} (token,seed) datapoints")

    print("[stage-d] loading new-token Stage C predictions…")
    b32_new = load_new_per_seed_per_token("b32")
    l14_new = load_new_per_seed_per_token("l14")
    print(f"  b32 new:      {sum(len(v) for v in b32_new.values())} (token,seed) datapoints")
    print(f"  l14 new:      {sum(len(v) for v in l14_new.values())} (token,seed) datapoints")

    print("[stage-d] writing per-token CSVs…")
    write_per_token_csv("size",  b32_existing, b32_new, l14_existing, l14_new)
    write_per_token_csv("speed", b32_existing, b32_new, l14_existing, l14_new)

    print("[stage-d] writing 3-axis class CSV…")
    write_3axis_csv(b32_existing, b32_new, l14_existing, l14_new)

    print("[stage-d] detecting surprise tokens…")
    surprises = detect_surprises(b32_existing, b32_new, l14_existing, l14_new)
    surp_out = TABLES / "surprises.json"
    surp_out.write_text(json.dumps(surprises, indent=2))
    print(f"  wrote {surp_out}")
    print("  --- per-class counts (B/32 outcome at thresholds 0.50 / 0.75) ---")
    for cls, c in surprises["class_counts"].items():
        print(f"    {cls:5s}  cliff_fail={c['cliff_fail']}  borderline={c['borderline']}  "
              f"ceiling_pass={c['ceiling_pass']}   agrees_expected={c['n_agree']}/{c['n_total']}")
    print(f"  --- {len(surprises['surprises'])} surprise(s) ---")
    for s in surprises["surprises"]:
        print(f"    {s['source']:5s}  {s['class']:5s}  {s['token']:11s}  "
              f"B/32={s['b32_mean']:.3f} ({s['b32_outcome']})  "
              f"L/14={s['l14_mean']:.3f}  ← {s['direction']}")

    # Dump raw lookup
    raw_out = TABLES / "per_seed_per_token.json"
    raw_out.write_text(json.dumps({
        "b32_existing": b32_existing, "b32_new": b32_new,
        "l14_existing": l14_existing, "l14_new": l14_new,
        "seeds": SEEDS, "existing_tokens": EXISTING_TOKENS,
        "new_tokens": NEW_TOKENS, "sem_class": SEM_CLASS,
    }, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x)))
    print(f"  wrote {raw_out}")


if __name__ == "__main__":
    main()
