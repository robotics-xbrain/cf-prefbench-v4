"""Qwen2-VL-2B judge on the v4 cliff test splits (Phase 5 Exp 5).

Adapted from scripts/run_qwen2vl_judge.py. Reads from data/cf_prefbench_v4/
and runs without swap consistency to save time (single call per row).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cf_pref_learning.utils.io import ensure_dir, read_jsonl, write_json, write_jsonl
from scripts.run_qwen2vl_judge import (
    PROMPT_TEMPLATE, _sample_frames, _make_grid_pil, _parse_answer,
)


SPLITS = [
    "test_heldout_lexical",
    "test_heldout_lexical_scoot",
    "test_heldout_size_miniature",
    "test_heldout_size_petite",
    "test_heldout_size_colossal",
    "test_heldout_size_gigantic",
    "test_heldout_speed_briskly",
    "test_heldout_speed_sluggishly",
    "test_heldout_speed_speedily",
    "test_heldout_speed_gradually",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/path/to/project")
    parser.add_argument("--model-path", default="/path/to/cache/qwen-vl/Qwen2-VL-2B-Instruct")
    parser.add_argument("--output-dir", default="outputs/phase5/exp5_vlm/qwen")
    parser.add_argument("--frames-per-video", type=int, default=8)
    args = parser.parse_args()
    root = Path(args.project_root)
    out_dir = root / args.output_dir
    ensure_dir(out_dir)

    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    print(f"loading Qwen2-VL from {args.model_path}", flush=True)
    t0 = time.time()
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    def ask(prompt, img_a, img_b):
        msgs = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img_a},
                {"type": "image", "image": img_b},
                {"type": "text", "text": prompt},
            ],
        }]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img_a, img_b], return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=200,
                                  do_sample=False, temperature=None, top_p=None)
        new_tokens = gen[:, inputs.input_ids.shape[1]:]
        out = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return out

    summary = {}
    for split in SPLITS:
        path = root / "data" / "cf_prefbench_v4" / f"{split}.jsonl"
        if not path.exists():
            print(f"  skip missing {split}")
            continue
        rows = read_jsonl(path)
        if split == "test_heldout_lexical":
            rows = [r for r in rows if r.get("axis") == "motion_sequence"]
        rows.sort(key=lambda r: r.get("example_id", ""))

        pred_path = out_dir / f"predictions_{split}.jsonl"
        existing = read_jsonl(pred_path) if pred_path.exists() else []
        already = {str(r.get("example_id")) for r in existing}
        new_predictions = list(existing)

        t_split = time.time()
        n_new = 0
        for i, row in enumerate(rows):
            ex_id = str(row["example_id"])
            if ex_id in already: continue
            try:
                fa = _sample_frames(root / row["video_a"], args.frames_per_video)
                fb = _sample_frames(root / row["video_b"], args.frames_per_video)
            except Exception as exc:
                new_predictions.append({
                    **row, "baseline": "qwen2-vl-2b", "prediction": "ERROR",
                    "prediction_score": 0.0, "error": f"frame_decode: {exc}",
                })
                continue
            img_a = _make_grid_pil(fa)
            img_b = _make_grid_pil(fb)
            prompt = PROMPT_TEMPLATE.format(instruction=row["instruction"])

            ans = None
            raw = ""
            try:
                raw = ask(prompt, img_a, img_b)
                ans = _parse_answer(raw)
            except Exception as exc:
                print(f"  ERROR {ex_id}: {exc}", flush=True)
                continue

            pred = ans if ans in {"A", "B"} else "Tie"
            new_predictions.append({
                "axis": row.get("axis"), "baseline": "qwen2-vl-2b",
                "counterfactual_flip_id": row.get("counterfactual_flip_id"),
                "example_id": ex_id, "instruction": row.get("instruction"),
                "pair_id": row.get("pair_id"),
                "paraphrase_group_id": row.get("paraphrase_group_id"),
                "prediction": pred,
                "prediction_score": 1.0 if pred == "A" else 0.0 if pred == "B" else 0.5,
                "preferred": row.get("preferred"),
                "split": split,
                "raw": raw,
            })
            n_new += 1
            if n_new % 10 == 0:
                write_jsonl(pred_path, new_predictions)
                elapsed = time.time() - t_split
                print(f"  [{split}] {n_new} new rows  elapsed={elapsed:.0f}s "
                      f"rate={n_new/elapsed:.2f}/s", flush=True)
        write_jsonl(pred_path, new_predictions)
        print(f"[{split}] DONE  n={len(new_predictions)}", flush=True)

    # Aggregate
    print("\n=== Aggregate per-split accuracy ===", flush=True)
    for split in SPLITS:
        pred_path = out_dir / f"predictions_{split}.jsonl"
        if not pred_path.exists(): continue
        preds = read_jsonl(pred_path)
        preds = [p for p in preds if p.get("prediction") in {"A", "B"}]
        if not preds: continue
        n_correct = sum(1 for p in preds if p["prediction"] == p["preferred"])
        accuracy = n_correct / len(preds)
        by_verb = defaultdict(list)
        for p in preds:
            by_verb[p["instruction"].split()[0]].append(p)
        per_verb = {v: sum(1 for p in rs if p["prediction"]==p["preferred"])/len(rs)
                    for v, rs in by_verb.items()}
        summary[split] = {"n": len(preds), "accuracy": accuracy, "per_verb": per_verb}
        print(f"  {split:40s} n={len(preds):3d}  acc={accuracy:.4f}", flush=True)
    write_json(out_dir / "summary.json", {"model": "qwen2-vl-2b", "per_split": summary})


if __name__ == "__main__":
    main()
