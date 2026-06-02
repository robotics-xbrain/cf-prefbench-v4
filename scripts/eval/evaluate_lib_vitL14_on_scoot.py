"""Evaluate a saved ViT-L/14 LIB v0 checkpoint on the scoot held-out test
+ regular motion held-out + size + speed held-out lexical splits.

Produces a 4-verb cliff curve under ViT-L/14 for direct comparison to B/32.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cf_pref_learning.utils.io import read_jsonl, write_json
from scripts.train_lib_v4_vitL14 import LIBSystemVitL14


def _encode_texts_L14(instructions: list[str]):
    os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:18080")
    os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:18080")
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="/path/to/cache/clip/ViT-L-14.pt"
    )
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    model = model.cuda().eval()
    with torch.no_grad():
        toks = tokenizer(instructions).cuda()
        feats = model.encode_text(toks).cpu().numpy().astype(np.float32)
    return feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed-label", required=True)
    parser.add_argument("--feature-path",
                        default="outputs/auto/v4_new_axes_features_vitL14_patches.npz")
    parser.add_argument("--scoot-jsonl",
                        default="data/cf_prefbench_v4/test_heldout_lexical_scoot.jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path("/path/to/project")

    data = np.load(root / args.feature_path, allow_pickle=True)
    video_patches = data["video_patches"].astype(np.float32)
    video_paths = list(data["video_paths"])
    video_to_idx = {v: i for i, v in enumerate(video_paths)}
    patch_dim = video_patches.shape[3]
    print(f"L/14 features shape: {video_patches.shape}  patch_dim={patch_dim}")

    rows = read_jsonl(root / args.scoot_jsonl)
    unique_instrs = sorted({r["instruction"] for r in rows})
    feats = _encode_texts_L14(unique_instrs)
    text_dim = feats.shape[1]
    instr_to_feat = {i: feats[k] for k, i in enumerate(unique_instrs)}

    sub_instrs = sorted({" ".join(["move"] + r["instruction"].split()[1:]) for r in rows})
    sub_feats = _encode_texts_L14(sub_instrs)
    sub_instr_to_feat = {i: sub_feats[k] for k, i in enumerate(sub_instrs)}

    dev = torch.device("cuda")
    system = LIBSystemVitL14(
        d_attr=128, dropout=0.3, n_attr=4,
        clip_text_dim=text_dim, clip_patch_dim=patch_dim,
    ).to(dev)
    state = torch.load(root / args.checkpoint, map_location=dev)
    system.load_state_dict(state)
    system.eval()

    video_patches_t = torch.from_numpy(video_patches).to(dev)

    def evaluate(rows, text_lookup):
        records = [r for r in rows
                    if r["video_a"] in video_to_idx and r["video_b"] in video_to_idx]
        if not records:
            return None, []
        scores = []
        with torch.no_grad():
            B = 8  # L/14 patches are bigger so smaller batch
            for s in range(0, len(records), B):
                chunk = records[s:s + B]
                va = torch.tensor([video_to_idx[r["video_a"]] for r in chunk],
                                    dtype=torch.long, device=dev)
                vb = torch.tensor([video_to_idx[r["video_b"]] for r in chunk],
                                    dtype=torch.long, device=dev)
                pa = video_patches_t[va]
                pb = video_patches_t[vb]
                txt = torch.from_numpy(
                    np.stack([text_lookup(r) for r in chunk]).astype(np.float32)
                ).to(dev)
                out = system(pa, pb, txt)
                scores.append(torch.sigmoid(out["score"]).cpu().numpy())
        return np.concatenate(scores), records

    normal_scores, recs = evaluate(rows, lambda r: instr_to_feat[r["instruction"]])
    normal_pred = ["A" if s >= 0.5 else "B" for s in normal_scores]
    normal_correct = sum(1 for p, r in zip(normal_pred, recs) if p == r["preferred"])

    sub_scores, _ = evaluate(rows, lambda r: sub_instr_to_feat[
        " ".join(["move"] + r["instruction"].split()[1:])
    ])
    sub_pred = ["A" if s >= 0.5 else "B" for s in sub_scores]
    sub_correct = sum(1 for p, r in zip(sub_pred, recs) if p == r["preferred"])

    by_dir = defaultdict(lambda: {"nc": 0, "sc": 0, "n": 0})
    for i, r in enumerate(recs):
        d = "L->R" if "left then right" in r["instruction"] else "R->L"
        by_dir[d]["n"] += 1
        if normal_pred[i] == r["preferred"]:
            by_dir[d]["nc"] += 1
        if sub_pred[i] == r["preferred"]:
            by_dir[d]["sc"] += 1

    summary = {
        "seed": args.seed_label,
        "checkpoint": args.checkpoint,
        "architecture": "ViT-L-14",
        "n_rows": len(recs),
        "normal": {
            "row_accuracy": normal_correct / len(recs),
            "n_correct": normal_correct,
            "per_direction": {d: {"acc": v["nc"]/v["n"], "n": v["n"]} for d, v in by_dir.items()},
        },
        "substitution": {
            "row_accuracy": sub_correct / len(recs),
            "n_correct": sub_correct,
            "per_direction": {d: {"acc": v["sc"]/v["n"], "n": v["n"]} for d, v in by_dir.items()},
        },
    }
    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, summary)
    print(f"\nseed {args.seed_label} ViT-L/14 scoot results:")
    print(f"  NORMAL: {normal_correct}/{len(recs)} = {normal_correct/len(recs):.4f}")
    print(f"  SUBSTITUTION: {sub_correct}/{len(recs)} = {sub_correct/len(recs):.4f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
