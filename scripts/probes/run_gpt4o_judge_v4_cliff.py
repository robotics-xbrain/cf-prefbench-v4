"""GPT-4o judge on the specific CF-PrefBench v4 cliff test splits.

Evaluates GPT-4o on motion_sequence test_heldout_lexical + scoot +
size new tokens + speed new tokens — the same conditions LIB v0 was
tested on in Exp 3, 3b, and 4. Lets us ask: does GPT-4o also show
the lexical cliff?
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cf_pref_learning.utils.io import ensure_dir, read_jsonl, write_json, write_jsonl
from scripts.run_gpt4o_judge import (
    PROMPT_TEMPLATE, _sample_frames, _make_grid, _img_b64, _parse_answer,
    _call_gpt4o, PRICE_IN_PER_TOKEN, PRICE_OUT_PER_TOKEN,
)


# v4 cliff test splits
SPLITS = [
    "test_heldout_lexical",        # 84 motion rows (shift/convey/transit) + 84 size + 84 speed; we filter
    "test_heldout_lexical_scoot",  # 28 rows
    "test_heldout_size_miniature", # 42 rows
    "test_heldout_size_petite",    # 42 rows
    "test_heldout_size_colossal",  # 42 rows
    "test_heldout_size_gigantic",  # 42 rows
    "test_heldout_speed_briskly",  # 42 rows
    "test_heldout_speed_sluggishly", # 42 rows
    "test_heldout_speed_speedily", # 42 rows
    "test_heldout_speed_gradually",# 42 rows
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/path/to/project")
    parser.add_argument("--output-dir", default="outputs/phase5/exp5_vlm/gpt4o")
    parser.add_argument("--model", default="gpt-4o-2024-11-20")
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--max-cost-usd", type=float, default=25.0)
    parser.add_argument("--no-swap", action="store_true", default=True,
                        help="default: skip swap-consistency for speed (halves cost)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--motion-axis-only", action="store_true",
                        help="for test_heldout_lexical, only keep motion_sequence rows")
    args = parser.parse_args()

    root = Path(args.project_root)
    out_dir = root / args.output_dir
    ensure_dir(out_dir)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(2)

    os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:18080")
    os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:18080")
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    total_cost = 0.0
    cost_log = []
    all_predictions = []

    for split in SPLITS:
        path = root / "data" / "cf_prefbench_v4" / f"{split}.jsonl"
        if not path.exists():
            print(f"  skip missing {split}")
            continue
        rows = read_jsonl(path)
        # Only keep motion_sequence rows from test_heldout_lexical (size+speed are in their own files)
        if split == "test_heldout_lexical":
            rows = [r for r in rows if r.get("axis") == "motion_sequence"]
        rows.sort(key=lambda r: r.get("example_id", ""))

        pred_path = out_dir / f"predictions_{split}.jsonl"
        already = set()
        existing = []
        if pred_path.exists():
            existing = read_jsonl(pred_path)
            already = {str(r.get("example_id")) for r in existing}

        new_predictions = list(existing)
        t_split = time.time()
        n_new = 0
        for i, row in enumerate(rows):
            ex_id = str(row["example_id"])
            if ex_id in already:
                continue
            if total_cost >= args.max_cost_usd:
                print(f"[cost cap] reached ${total_cost:.2f}; stopping", flush=True)
                break
            try:
                fa = _sample_frames(root / row["video_a"], args.frames_per_video)
                fb = _sample_frames(root / row["video_b"], args.frames_per_video)
            except Exception as exc:
                new_predictions.append({
                    **row, "baseline": args.model,
                    "prediction": "ERROR", "prediction_score": 0.0,
                    "error": f"frame_decode: {exc}",
                })
                continue
            img_a = _img_b64(_make_grid(fa))
            img_b = _img_b64(_make_grid(fb))
            prompt = PROMPT_TEMPLATE.format(instruction=row["instruction"])

            ans_orig = None
            usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            raw_orig = ""
            try:
                raw_orig, usage = _call_gpt4o(client, args.model, prompt, img_a, img_b, args.temperature)
                ans_orig = _parse_answer(raw_orig)
                for k in usage_total:
                    usage_total[k] += usage[k]
            except Exception as exc:
                print(f"  ERROR {ex_id}: {exc}", flush=True)
                continue

            pred = ans_orig if ans_orig in {"A", "B"} else "Tie"
            cost_call = (usage_total["prompt_tokens"] * PRICE_IN_PER_TOKEN
                         + usage_total["completion_tokens"] * PRICE_OUT_PER_TOKEN)
            total_cost += cost_call
            cost_log.append({"example_id": ex_id, "split": split, "cost": cost_call})

            new_predictions.append({
                "axis": row.get("axis"),
                "baseline": args.model,
                "counterfactual_flip_id": row.get("counterfactual_flip_id"),
                "example_id": ex_id,
                "instruction": row.get("instruction"),
                "pair_id": row.get("pair_id"),
                "paraphrase_group_id": row.get("paraphrase_group_id"),
                "prediction": pred,
                "prediction_score": 1.0 if pred == "A" else 0.0 if pred == "B" else 0.5,
                "preferred": row.get("preferred"),
                "split": split,
                "raw_orig": raw_orig,
                "ans_orig": ans_orig,
                "usage": usage_total,
                "cost_usd": cost_call,
            })
            n_new += 1
            if n_new % 10 == 0:
                write_jsonl(pred_path, new_predictions)
                elapsed = time.time() - t_split
                print(f"  [{split}] {n_new} new rows  cost=${total_cost:.2f}  "
                      f"elapsed={elapsed:.0f}s  rate={n_new/elapsed:.2f}/s", flush=True)
        write_jsonl(pred_path, new_predictions)
        all_predictions.extend(new_predictions)
        print(f"[{split}] DONE  n={len(new_predictions)}  cost-so-far=${total_cost:.2f}", flush=True)

    write_json(out_dir / "cost_log.json", {"total_cost_usd": total_cost, "per_call": cost_log})

    # Per-split per-verb accuracy
    print("\n=== Aggregate per-split accuracy ===", flush=True)
    summary = {}
    for split in SPLITS:
        pred_path = out_dir / f"predictions_{split}.jsonl"
        if not pred_path.exists(): continue
        preds = read_jsonl(pred_path)
        preds = [p for p in preds if p.get("prediction") in {"A", "B"}]
        if not preds: continue
        n_correct = sum(1 for p in preds if p["prediction"] == p["preferred"])
        accuracy = n_correct / len(preds)
        # Per-verb breakdown
        by_verb = defaultdict(list)
        for p in preds:
            v = p["instruction"].split()[0]
            by_verb[v].append(p)
        per_verb = {}
        for v, rs in by_verb.items():
            nc = sum(1 for p in rs if p["prediction"] == p["preferred"])
            per_verb[v] = {"acc": nc/len(rs), "n": len(rs)}
        summary[split] = {"n": len(preds), "accuracy": accuracy, "per_verb": per_verb}
        verb_str = ", ".join(f"{v}={vi['acc']:.3f}" for v, vi in per_verb.items())
        print(f"  {split:40s} n={len(preds):3d}  acc={accuracy:.4f}  per_verb: {verb_str}", flush=True)

    write_json(out_dir / "summary.json", {
        "model": args.model, "total_cost_usd": total_cost, "per_split": summary,
    })
    print(f"\nTotal cost: ${total_cost:.2f}  wrote {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
