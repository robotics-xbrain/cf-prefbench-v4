"""Evaluate a saved LIB v0 checkpoint on the size/speed axis-generalization
test data (Phase 5 Exp 4).

For each new token (e.g., colossal, miniature, briskly, sluggishly):
  - CLIP-encode the new instruction text
  - Run LIB v0 forward on the (video_a, video_b, new_text) triples
  - Report per-token per-direction accuracy
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cf_pref_learning.utils.io import read_jsonl, write_json
from scripts.train_lib import LIBSystem
from scripts.train_lib_v4_vitL14 import LIBSystemVitL14


def _encode_texts(instructions, model_name, pretrained):
    os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:18080")
    os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:18080")
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.cuda().eval()
    with torch.no_grad():
        toks = tokenizer(instructions).cuda()
        feats = model.encode_text(toks).cpu().numpy().astype(np.float32)
    return feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature-path", required=True)
    parser.add_argument("--arch", choices=["B32", "L14"], required=True)
    parser.add_argument("--seed-label", required=True)
    parser.add_argument("--test-files", nargs="+", required=True,
                        help="test_heldout_*.jsonl filenames under data/cf_prefbench_v4/")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path("/path/to/project")

    data = np.load(root / args.feature_path, allow_pickle=True)
    video_patches = data["video_patches"].astype(np.float32)
    video_paths = list(data["video_paths"])
    video_to_idx = {v: i for i, v in enumerate(video_paths)}
    patch_dim = video_patches.shape[3]

    dev = torch.device("cuda")
    video_patches_t = torch.from_numpy(video_patches).to(dev)

    # Build the appropriate system
    if args.arch == "B32":
        system = LIBSystem(d_attr=128, dropout=0.3, n_attr=4).to(dev)
    else:
        system = LIBSystemVitL14(d_attr=128, dropout=0.3, n_attr=4,
                                   clip_text_dim=768, clip_patch_dim=1024).to(dev)
    state = torch.load(root / args.checkpoint, map_location=dev)
    system.load_state_dict(state)
    system.eval()

    model_name = "ViT-B-32" if args.arch == "B32" else "ViT-L-14"
    pretrained = (
        "/path/to/cache/clip/ViT-B-32.pt" if args.arch == "B32"
        else "/path/to/cache/clip/ViT-L-14.pt"
    )

    out_summary = {"checkpoint": args.checkpoint, "arch": args.arch,
                    "seed_label": args.seed_label, "results": {}}

    for tf in args.test_files:
        rows = read_jsonl(root / "data" / "cf_prefbench_v4" / tf)
        # Filter to rows whose videos are in the feature index
        rows = [r for r in rows if r["video_a"] in video_to_idx and r["video_b"] in video_to_idx]
        if not rows:
            print(f"  {tf}: NO ROWS in feature index (skipping)")
            continue

        unique_instrs = sorted({r["instruction"] for r in rows})
        feats = _encode_texts(unique_instrs, model_name, pretrained)
        i2f = {ins: feats[k] for k, ins in enumerate(unique_instrs)}

        scores = []
        with torch.no_grad():
            B = 16 if args.arch == "B32" else 8
            for s in range(0, len(rows), B):
                chunk = rows[s:s + B]
                va = torch.tensor([video_to_idx[r["video_a"]] for r in chunk],
                                    dtype=torch.long, device=dev)
                vb = torch.tensor([video_to_idx[r["video_b"]] for r in chunk],
                                    dtype=torch.long, device=dev)
                pa = video_patches_t[va]
                pb = video_patches_t[vb]
                txt = torch.from_numpy(
                    np.stack([i2f[r["instruction"]] for r in chunk]).astype(np.float32)
                ).to(dev)
                out = system(pa, pb, txt)
                scores.append(torch.sigmoid(out["score"]).cpu().numpy())
        scores = np.concatenate(scores)
        pred = ["A" if s >= 0.5 else "B" for s in scores]
        n_correct = sum(1 for p, r in zip(pred, rows) if p == r["preferred"])

        # Per-paraphrase-verb (in size/speed, the verb is what's varied in train held-out)
        by_verb = defaultdict(list)
        for i, r in enumerate(rows):
            v = r["instruction"].split()[0]
            by_verb[v].append((pred[i], r["preferred"]))
        per_verb = {v: sum(1 for p, t in items if p == t) / len(items) for v, items in by_verb.items()}

        out_summary["results"][tf] = {
            "n": len(rows),
            "accuracy": n_correct / len(rows),
            "n_correct": n_correct,
            "per_verb": per_verb,
        }
        print(f"  {tf:40s} n={len(rows)}  acc={n_correct/len(rows):.4f}  per_verb={ {k: round(v,3) for k,v in per_verb.items()} }")

    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, out_summary)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
