"""Per-frame CLIP patch token extractor.

For each unique video in CF-PrefBench v3, sample K=8 frames, run them
through CLIP ViT-B/32 image encoder with `output_tokens=True`, and
save the resulting (K, 49, 768) patch tokens. Also encode the
unique instructions to get text features.

Output: outputs/auto/v3_features_clip_patches.npz
    video_patches: (N, K, 49, 768) float16  (≈ 14 GB on disk
                                              compressed)
    text_features:  (N, 512) float32
    example_ids:    (N,) object

Float16 storage is critical — float32 would be ~28 GB and exceed the
500 GB available on /data3. The downstream LIB module casts back to
float32 inside the training loop.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cf_pref_learning.utils.io import ensure_dir, read_jsonl, write_json
from scripts.extract_frame_clip_features import (
    SPLITS, _load_rows, _decode_video_frames, _encode_texts,
)


def _encode_video_patches(model, preprocess, frames, device: str) -> np.ndarray:
    """Returns [K, 49, 768] float16."""
    import torch
    from PIL import Image
    pil = [Image.fromarray(f) for f in frames]
    batch = torch.stack([preprocess(im) for im in pil], dim=0).to(device)
    with torch.no_grad():
        # model.visual.output_tokens was set in main() before this call
        pooled, tokens = model.visual(batch)
    return tokens.detach().cpu().to(torch.float16).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/path/to/project")
    parser.add_argument("--output", default="outputs/auto/v3_features_clip_patches.npz")
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained",
                        default="/path/to/cache/clip/ViT-B-32.pt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:18080")
    os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:18080")
    import open_clip
    import torch

    root = Path(args.project_root)
    out_path = root / args.output
    ensure_dir(out_path.parent)
    rows = _load_rows(root)
    videos = sorted({r["video_a"] for r in rows} | {r["video_b"] for r in rows})
    instructions = sorted({r["instruction"] for r in rows})
    print(f"loaded {len(rows)} rows; {len(videos)} unique videos; {len(instructions)} unique instructions", flush=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name, pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model = model.to(device).eval()
    model.visual.output_tokens = True  # expose patch tokens

    K = args.frames_per_video
    perframe: dict[str, np.ndarray] = {}
    t0 = time.time()
    for i, rel in enumerate(videos):
        vp = Path(rel)
        if not vp.is_absolute():
            vp = root / vp
        try:
            frames = _decode_video_frames(vp, K)
            perframe[rel] = _encode_video_patches(model, preprocess, frames, device)
        except Exception as exc:
            print(f"  WARN: {rel}: {exc}", flush=True)
            perframe[rel] = np.zeros((K, 49, 768), dtype=np.float16)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  videos {i+1}/{len(videos)}  rate={(i+1)/elapsed:.2f}/s", flush=True)

    # Encode text (existing utility produces float32)
    instr_feats = _encode_texts(model, tokenizer, instructions, device)
    instr_to_idx = {s: i for i, s in enumerate(instructions)}

    # Assemble per-row features. Each row references TWO videos, so
    # storing per-row patches (2K frames per row) blows up memory.
    # Instead, store per-VIDEO patches once and a (video_a_idx, video_b_idx)
    # lookup at training time.
    video_to_idx = {v: i for i, v in enumerate(videos)}
    video_patches_arr = np.stack([perframe[v] for v in videos], axis=0)  # [n_videos, K, 49, 768]

    row_keys = [str(row["example_id"]) for row in rows]
    row_video_a = np.array([video_to_idx[row["video_a"]] for row in rows], dtype=np.int32)
    row_video_b = np.array([video_to_idx[row["video_b"]] for row in rows], dtype=np.int32)
    row_text = np.stack([instr_feats[instr_to_idx[row["instruction"]]] for row in rows], axis=0)

    np.savez_compressed(
        out_path,
        video_patches=video_patches_arr,         # float16 [n_videos, K, 49, 768]
        text_features=row_text,                  # float32 [n_rows, 512]
        row_video_a_idx=row_video_a,             # int32 [n_rows]
        row_video_b_idx=row_video_b,             # int32 [n_rows]
        example_ids=np.array(row_keys, dtype=object),
        video_paths=np.array(videos, dtype=object),
    )

    meta = {
        "n_rows": len(rows),
        "n_videos": len(videos),
        "video_patches_shape": list(video_patches_arr.shape),
        "video_patches_dtype": "float16",
        "patch_count_per_frame": 49,
        "patch_dim": 768,
        "K": K,
        "model": args.model_name,
        "pretrained": args.pretrained,
        "purpose": "LIB training input: patch-level CLIP features for cross-attention.",
    }
    write_json(root / args.output.replace(".npz", "_meta.json"), meta)
    print(f"wrote {out_path}  patches={video_patches_arr.shape} text={row_text.shape}", flush=True)


if __name__ == "__main__":
    main()
