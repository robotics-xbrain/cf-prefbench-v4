"""Extract CLIP ViT-L/14 patch features for v4 NEW axes only.

ViT-L/14 has 256 patches per 224×224 image and a 1024-d patch dim.
The training pipeline (LIB v0) is patch-dim-agnostic; the cross-attention
projects from clip-text-dim → n_attr*d_attr and attends over patches with
their native dim. We just need to plumb the patch feature shape through.

Output: outputs/auto/v4_new_axes_features_vitL14_patches.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cf_pref_learning.utils.io import ensure_dir, read_jsonl, write_json
from scripts.extract_frame_clip_features import (
    _decode_video_frames, _encode_texts,
)


NEW_AXES = {"size", "motion_sequence", "speed"}
SPLITS = [
    "train", "val", "test_seen",
    "test_heldout_lexical", "test_heldout_camera",
    "test_heldout_color", "test_heldout_spatial",
    "test_hard_negatives",
]


def _load_new_axis_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s in SPLITS:
        for r in read_jsonl(root / "data" / "cf_prefbench_v4" / f"{s}.jsonl"):
            if r["axis"] in NEW_AXES:
                rows.append(r)
    return rows


def _encode_video_patches(model, preprocess, frames, device):
    import torch
    from PIL import Image
    pil = [Image.fromarray(f) for f in frames]
    batch = torch.stack([preprocess(im) for im in pil], dim=0).to(device)
    with torch.no_grad():
        pooled, tokens = model.visual(batch)
    return tokens.detach().cpu().to(torch.float16).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/path/to/project")
    parser.add_argument("--output", default="outputs/auto/v4_new_axes_features_vitL14_patches.npz")
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--model-name", default="ViT-L-14")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:18080")
    os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:18080")
    import open_clip
    import torch

    root = Path(args.project_root)
    out_path = root / args.output
    ensure_dir(out_path.parent)

    rows = _load_new_axis_rows(root)
    videos = sorted({r["video_a"] for r in rows} | {r["video_b"] for r in rows})
    instructions = sorted({r["instruction"] for r in rows})
    print(f"loaded {len(rows)} new-axis rows; {len(videos)} videos; "
          f"{len(instructions)} instructions", flush=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"loading {args.model_name} / {args.pretrained}", flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name, pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model = model.to(device).eval()
    model.visual.output_tokens = True

    K = args.frames_per_video
    perframe = {}
    t0 = time.time()
    for i, rel in enumerate(videos):
        vp = Path(rel)
        if not vp.is_absolute():
            vp = root / vp
        try:
            frames = _decode_video_frames(vp, K)
            perframe[rel] = _encode_video_patches(model, preprocess, frames, device)
        except Exception as exc:
            print(f"  WARN {rel}: {exc}", flush=True)
            perframe[rel] = np.zeros((K, 256, 1024), dtype=np.float16)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  videos {i+1}/{len(videos)}  rate={(i+1)/elapsed:.2f}/s", flush=True)
    print(f"video encoding done in {time.time()-t0:.1f}s", flush=True)

    instr_feats = _encode_texts(model, tokenizer, instructions, device)
    instr_to_idx = {s: i for i, s in enumerate(instructions)}
    video_to_idx = {v: i for i, v in enumerate(videos)}
    video_patches_arr = np.stack([perframe[v] for v in videos], axis=0)

    row_keys = [str(r["example_id"]) for r in rows]
    row_video_a = np.array([video_to_idx[r["video_a"]] for r in rows], dtype=np.int32)
    row_video_b = np.array([video_to_idx[r["video_b"]] for r in rows], dtype=np.int32)
    row_text = np.stack([instr_feats[instr_to_idx[r["instruction"]]] for r in rows], axis=0)

    np.savez_compressed(
        out_path,
        video_patches=video_patches_arr,
        text_features=row_text,
        row_video_a_idx=row_video_a,
        row_video_b_idx=row_video_b,
        example_ids=np.array(row_keys, dtype=object),
        video_paths=np.array(videos, dtype=object),
    )
    meta = {
        "n_rows": len(rows),
        "n_videos": len(videos),
        "video_patches_shape": list(video_patches_arr.shape),
        "video_patches_dtype": "float16",
        "patch_count_per_frame": int(video_patches_arr.shape[2]),
        "patch_dim": int(video_patches_arr.shape[3]),
        "text_dim": int(row_text.shape[1]),
        "K": K,
        "model": args.model_name,
        "pretrained": args.pretrained,
        "axes_included": sorted(NEW_AXES),
    }
    write_json(root / args.output.replace(".npz", "_meta.json"), meta)
    print(f"wrote {out_path}  patches={video_patches_arr.shape} text={row_text.shape}", flush=True)


if __name__ == "__main__":
    main()
