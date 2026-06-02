"""SC-6 sanity probe for VLM judges: shuffle instruction text within
the test split and measure accuracy drop.

For each row, replace its instruction with another row's instruction
from the same split (cyclic shift). Run the VLM judge as normal.
If the model uses the instruction, accuracy should drop. If accuracy
stays at NORMAL, the model is not using the instruction.

Supports two backends: --backend gpt4o or --backend qwen.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cf_pref_learning.utils.io import ensure_dir, read_jsonl, write_json, write_jsonl


SPLITS_MOTION = ["test_heldout_lexical"]  # motion-axis subset is the key cliff signal


def _shuffle_instructions(rows: list[dict], seed: int = 0):
    """Replace each row's instruction with a DIFFERENT instruction from the pool.

    Critical: the v4 motion held-out has only 6 unique instructions (3 verbs × 2 directions);
    we must ensure the replacement is a string-different instruction, ideally with the
    OPPOSITE direction tokens so the test actually breaks the binding.
    """
    rng = np.random.default_rng(seed)
    out = []
    instr_pool = list({r["instruction"] for r in rows})
    for r in rows:
        # Pick a different instruction
        candidates = [x for x in instr_pool if x != r["instruction"]]
        # Prefer one with opposite direction tokens if possible
        orig_dirs = tuple(w for w in r["instruction"].split() if w in {"left","right","up","down"})
        opp_candidates = [c for c in candidates
                          if tuple(w for w in c.split() if w in {"left","right","up","down"}) != orig_dirs]
        pool = opp_candidates if opp_candidates else candidates
        chosen = pool[rng.integers(len(pool))]
        new_r = dict(r)
        new_r["instruction"] = chosen
        new_r["original_instruction"] = r["instruction"]
        out.append(new_r)
    return out


def _run_gpt4o(rows, root, output_dir, max_cost=2.0):
    from scripts.run_gpt4o_judge import (
        PROMPT_TEMPLATE, _sample_frames, _make_grid, _img_b64,
        _parse_answer, _call_gpt4o, PRICE_IN_PER_TOKEN, PRICE_OUT_PER_TOKEN,
    )
    os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:18080")
    os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:18080")
    from openai import OpenAI
    client = OpenAI()
    predictions = []
    total_cost = 0.0
    for i, row in enumerate(rows):
        if total_cost >= max_cost:
            print(f"  cost cap reached ${total_cost:.2f}", flush=True)
            break
        try:
            fa = _sample_frames(root / row["video_a"], 8)
            fb = _sample_frames(root / row["video_b"], 8)
        except Exception as exc:
            predictions.append({**row, "prediction": "ERROR", "error": str(exc)})
            continue
        img_a = _img_b64(_make_grid(fa))
        img_b = _img_b64(_make_grid(fb))
        prompt = PROMPT_TEMPLATE.format(instruction=row["instruction"])
        try:
            raw, usage = _call_gpt4o(client, "gpt-4o-2024-11-20", prompt, img_a, img_b, 0.0)
            ans = _parse_answer(raw)
        except Exception as exc:
            print(f"  ERROR {row['example_id']}: {exc}", flush=True)
            continue
        pred = ans if ans in {"A", "B"} else "Tie"
        cost = usage["prompt_tokens"] * PRICE_IN_PER_TOKEN + usage["completion_tokens"] * PRICE_OUT_PER_TOKEN
        total_cost += cost
        predictions.append({
            "example_id": row["example_id"],
            "video_a": row["video_a"], "video_b": row["video_b"],
            "shuffled_instruction": row["instruction"],
            "original_instruction": row["original_instruction"],
            "preferred": row["preferred"],
            "prediction": pred,
            "prediction_score": 1.0 if pred == "A" else 0.0 if pred == "B" else 0.5,
            "raw": raw,
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(rows)}  cost=${total_cost:.2f}", flush=True)
    return predictions, total_cost


def _run_qwen(rows, root, output_dir):
    from scripts.run_qwen2vl_judge import (
        PROMPT_TEMPLATE, _sample_frames, _make_grid_pil, _parse_answer,
    )
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    model_path = "/path/to/cache/qwen-vl/Qwen2-VL-2B-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()
    print(f"  Qwen loaded", flush=True)
    predictions = []
    for i, row in enumerate(rows):
        try:
            fa = _sample_frames(root / row["video_a"], 8)
            fb = _sample_frames(root / row["video_b"], 8)
        except Exception as exc:
            predictions.append({**row, "prediction": "ERROR", "error": str(exc)})
            continue
        img_a = _make_grid_pil(fa)
        img_b = _make_grid_pil(fb)
        prompt = PROMPT_TEMPLATE.format(instruction=row["instruction"])
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img_a},
            {"type": "image", "image": img_b},
            {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img_a, img_b], return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=200, do_sample=False,
                                 temperature=None, top_p=None)
        nt = gen[:, inputs.input_ids.shape[1]:]
        out = processor.batch_decode(nt, skip_special_tokens=True)[0]
        ans = _parse_answer(out)
        pred = ans if ans in {"A", "B"} else "Tie"
        predictions.append({
            "example_id": row["example_id"],
            "video_a": row["video_a"], "video_b": row["video_b"],
            "shuffled_instruction": row["instruction"],
            "original_instruction": row["original_instruction"],
            "preferred": row["preferred"],
            "prediction": pred,
            "prediction_score": 1.0 if pred == "A" else 0.0 if pred == "B" else 0.5,
            "raw": out,
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)
    return predictions, 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["gpt4o", "qwen"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--max-cost-usd", type=float, default=2.0)
    args = parser.parse_args()
    root = Path("/path/to/project")
    out_dir = root / args.output_dir
    ensure_dir(out_dir)

    # Load motion held-out lexical rows
    motion_rows = []
    for split in SPLITS_MOTION:
        path = root / "data" / "cf_prefbench_v4" / f"{split}.jsonl"
        rows = read_jsonl(path)
        motion_rows.extend([r for r in rows if r.get("axis") == "motion_sequence"])
    print(f"loaded {len(motion_rows)} motion held-out rows", flush=True)

    # Also include the scoot rows
    for split in ["test_heldout_lexical_scoot"]:
        path = root / "data" / "cf_prefbench_v4" / f"{split}.jsonl"
        if path.exists():
            motion_rows.extend(read_jsonl(path))
    print(f"total motion + scoot rows: {len(motion_rows)}", flush=True)

    shuffled = _shuffle_instructions(motion_rows, seed=args.shuffle_seed)
    print(f"shuffled instructions; sample: {shuffled[0]['original_instruction']!r} -> {shuffled[0]['instruction']!r}", flush=True)

    if args.backend == "gpt4o":
        preds, cost = _run_gpt4o(shuffled, root, out_dir, args.max_cost_usd)
    else:
        preds, cost = _run_qwen(shuffled, root, out_dir)

    out_path = out_dir / f"sc6_shuffle_{args.backend}.jsonl"
    write_jsonl(out_path, preds)

    # Accuracy summary
    n = len([p for p in preds if p.get("prediction") in {"A", "B"}])
    nc = sum(1 for p in preds if p.get("prediction") == p.get("preferred"))
    print(f"\nSC-6 (shuffle) {args.backend} accuracy: {nc}/{n} = {nc/n:.4f}")
    print(f"Cost: ${cost:.2f}")

    write_json(out_dir / f"sc6_summary_{args.backend}.json", {
        "backend": args.backend, "n": n, "n_correct": nc,
        "accuracy": nc/n if n else None, "cost_usd": cost,
        "n_motion_rows_total": len(motion_rows),
    })


if __name__ == "__main__":
    main()
