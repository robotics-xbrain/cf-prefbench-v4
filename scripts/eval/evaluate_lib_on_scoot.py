"""Evaluate a saved LIB v0 checkpoint on the scoot held-out test split.

Steps:
  1. Load existing v4 features npz (provides video patches indexed by video path)
  2. CLIP-B/32 encode "scoot the block ..." instructions
  3. For each pair_id in scoot test, look up its videos and the matching
     text feature; run LIB v0 forward and compute prediction
  4. Report per-row accuracy, A/B distribution, per-direction breakdown

Also runs SUBSTITUTION sanity test (replace scoot text with "move..."
train-verb text) for direct comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cf_pref_learning.utils.io import read_jsonl, write_json
from scripts.train_lib import LIBSystem


def _encode_texts(instructions: list[str]):
    os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:18080")
    os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:18080")
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="/path/to/cache/clip/ViT-B-32.pt"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.cuda().eval()
    with torch.no_grad():
        toks = tokenizer(instructions).cuda()
        feats = model.encode_text(toks).cpu().numpy().astype(np.float32)
    return feats  # NOT L2-normalized (matches training pipeline)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed-label", required=True)
    parser.add_argument("--feature-path",
                        default="outputs/auto/v4_new_axes_features_clip_patches.npz")
    parser.add_argument("--scoot-jsonl",
                        default="data/cf_prefbench_v4/test_heldout_lexical_scoot.jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path("/path/to/project")

    data = np.load(root / args.feature_path, allow_pickle=True)
    video_patches = data["video_patches"].astype(np.float32)
    video_paths = list(data["video_paths"])
    video_to_idx = {v: i for i, v in enumerate(video_paths)}

    rows = read_jsonl(root / args.scoot_jsonl)
    print(f"loaded {len(rows)} scoot rows")

    # Encode the scoot instructions
    unique_instrs = sorted({r["instruction"] for r in rows})
    feats = _encode_texts(unique_instrs)
    instr_to_feat = {i: feats[k] for k, i in enumerate(unique_instrs)}

    # Also encode train-verb substitutions for sanity test
    sub_instrs = sorted({" ".join(["move"] + r["instruction"].split()[1:]) for r in rows})
    sub_feats = _encode_texts(sub_instrs)
    sub_instr_to_feat = {i: sub_feats[k] for k, i in enumerate(sub_instrs)}

    dev = torch.device("cuda")
    system = LIBSystem(d_attr=128, dropout=0.3, n_attr=4).to(dev)
    state = torch.load(root / args.checkpoint, map_location=dev)
    system.load_state_dict(state)
    system.eval()

    video_patches_t = torch.from_numpy(video_patches).to(dev)

    def evaluate(rows, text_lookup):
        records = []
        for r in rows:
            if r["video_a"] not in video_to_idx or r["video_b"] not in video_to_idx:
                continue
            records.append(r)
        if not records:
            return None, []

        # Run inference
        scores = []
        with torch.no_grad():
            B = 32
            for s in range(0, len(records), B):
                chunk = records[s:s + B]
                va_idx = torch.tensor([video_to_idx[r["video_a"]] for r in chunk],
                                       dtype=torch.long, device=dev)
                vb_idx = torch.tensor([video_to_idx[r["video_b"]] for r in chunk],
                                       dtype=torch.long, device=dev)
                pa = video_patches_t[va_idx]
                pb = video_patches_t[vb_idx]
                txt = torch.from_numpy(
                    np.stack([text_lookup(r) for r in chunk]).astype(np.float32)
                ).to(dev)
                out = system(pa, pb, txt)
                scores.append(torch.sigmoid(out["score"]).cpu().numpy())
        scores = np.concatenate(scores)
        return scores, records

    # NORMAL: scoot instructions
    normal_scores, recs = evaluate(rows, lambda r: instr_to_feat[r["instruction"]])
    normal_pred = ["A" if s >= 0.5 else "B" for s in normal_scores]
    normal_correct = sum(1 for p, r in zip(normal_pred, recs) if p == r["preferred"])

    # SUBSTITUTION: scoot → move (train verb)
    sub_scores, _ = evaluate(rows, lambda r: sub_instr_to_feat[
        " ".join(["move"] + r["instruction"].split()[1:])
    ])
    sub_pred = ["A" if s >= 0.5 else "B" for s in sub_scores]
    sub_correct = sum(1 for p, r in zip(sub_pred, recs) if p == r["preferred"])

    # Per-direction breakdown
    by_dir = defaultdict(lambda: {"normal_correct": 0, "n": 0, "sub_correct": 0})
    for i, r in enumerate(recs):
        d = "L->R" if "left then right" in r["instruction"] else "R->L"
        by_dir[d]["n"] += 1
        if normal_pred[i] == r["preferred"]:
            by_dir[d]["normal_correct"] += 1
        if sub_pred[i] == r["preferred"]:
            by_dir[d]["sub_correct"] += 1

    summary = {
        "seed": args.seed_label,
        "checkpoint": args.checkpoint,
        "n_rows": len(recs),
        "normal": {
            "row_accuracy": normal_correct / len(recs),
            "row_correct": normal_correct,
            "scores_min": float(np.min(normal_scores)),
            "scores_max": float(np.max(normal_scores)),
            "scores_mean": float(np.mean(normal_scores)),
            "n_pred_A": int(sum(1 for p in normal_pred if p == "A")),
            "n_pred_B": int(sum(1 for p in normal_pred if p == "B")),
            "per_direction": {d: {"acc": v["normal_correct"] / v["n"], "n": v["n"]} for d, v in by_dir.items()},
        },
        "substitution_with_move": {
            "row_accuracy": sub_correct / len(recs),
            "row_correct": sub_correct,
            "per_direction": {d: {"acc": v["sub_correct"] / v["n"], "n": v["n"]} for d, v in by_dir.items()},
        },
    }

    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, summary)
    print(f"\nseed {args.seed_label} scoot results:")
    print(f"  NORMAL (scoot text): acc = {normal_correct}/{len(recs)} = {normal_correct/len(recs):.4f}")
    for d, v in by_dir.items():
        print(f"    {d}: {v['normal_correct']}/{v['n']} = {v['normal_correct']/v['n']:.4f}")
    print(f"  SUBSTITUTION (scoot→move): acc = {sub_correct}/{len(recs)} = {sub_correct/len(recs):.4f}")
    for d, v in by_dir.items():
        print(f"    {d}: {v['sub_correct']}/{v['n']} = {v['sub_correct']/v['n']:.4f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
