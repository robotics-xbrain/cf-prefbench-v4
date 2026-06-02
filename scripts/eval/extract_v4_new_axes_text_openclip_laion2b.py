"""EXP-H: extract OpenCLIP-LAION-2B text features for all v4 new-axis rows.

Mirrors the example_id ordering of the existing OpenAI-encoded text NPZ at
`outputs/auto/v4_new_axes_features_clip_patches.npz` so the trainer's
`text_npz` slot can swap in without any other change.

Output: `outputs/auto/v4_new_axes_features_openclip_b32_laion2b.npz`
  keys: example_ids[N], text_features[N, 512] fp32

Encoder: open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
Tokenizer: open_clip.get_tokenizer("ViT-B-32")   # same SimpleTokenizer that OpenAI uses
Batch size: 256
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/path/to/project")
sys.path.insert(0, str(ROOT))

from cf_pref_learning.utils.io import read_jsonl  # noqa: E402

NEW_AXES = {"size", "motion_sequence", "speed"}
SPLITS = [
    "train", "val", "test_seen",
    "test_heldout_lexical", "test_heldout_camera",
    "test_heldout_color", "test_heldout_spatial",
    "test_hard_negatives",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",
                        default=str(ROOT / "outputs/auto/v4_new_axes_features_openclip_b32_laion2b.npz"))
    parser.add_argument("--ref-npz",
                        default=str(ROOT / "outputs/auto/v4_new_axes_features_clip_patches.npz"),
                        help="Reference NPZ whose example_id ordering we mirror.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    # ---- Load reference ordering ----
    ref = np.load(args.ref_npz, allow_pickle=True)
    ref_example_ids = list(ref["example_ids"])
    print(f"  reference NPZ: {args.ref_npz}")
    print(f"  reference example_ids count: {len(ref_example_ids)}")

    # ---- Collect (example_id, instruction) for the new-axis rows from splits ----
    eid_to_instr = {}
    for s in SPLITS:
        for r in read_jsonl(ROOT / f"data/cf_prefbench_v4/{s}.jsonl"):
            if r["axis"] in NEW_AXES:
                eid = str(r["example_id"])
                eid_to_instr[eid] = r["instruction"]
    print(f"  collected {len(eid_to_instr)} unique (example_id → instruction) pairs from new-axes splits")

    # Verify reference IDs are a subset of what we have
    missing = [e for e in ref_example_ids if e not in eid_to_instr]
    if missing:
        sys.exit(f"ERROR: {len(missing)} reference example_ids missing from collected rows. "
                 f"First few: {missing[:5]}")

    instructions = [eid_to_instr[e] for e in ref_example_ids]
    print(f"  ordered instruction list length: {len(instructions)}")

    # ---- Load OpenCLIP LAION-2B B/32 ----
    os.environ.setdefault("HF_HOME", "/path/to/cache/huggingface")
    import open_clip
    print(f"\n  loading open_clip ViT-B-32 laion2b_s34b_b79k…")
    t0 = time.time()
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    # FREEZE: the model is used inference-only; no params will be trained.
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(args.device).eval()
    print(f"  loaded + frozen in {time.time() - t0:.1f}s")

    # ---- Encode in batches of 256, fp32 output ----
    feats_all = []
    print(f"\n  encoding {len(instructions)} instructions, batch={args.batch_size}…")
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(instructions), args.batch_size):
            chunk = instructions[s:s + args.batch_size]
            toks = tokenizer(chunk).to(args.device)
            feats = model.encode_text(toks)
            feats_all.append(feats.float().cpu().numpy())
    text_features = np.concatenate(feats_all, axis=0).astype(np.float32)
    print(f"  encoded in {time.time() - t0:.1f}s; shape={text_features.shape} dtype={text_features.dtype}")

    # ---- Save ----
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        example_ids=np.array(ref_example_ids, dtype=object),
        text_features=text_features,
        # Pass through patch-side fields so this NPZ is self-contained for
        # any downstream tool that expects the same schema as the OpenAI NPZ.
        # Note: video_patches are NOT duplicated (avoid 5 GB bloat); only the
        # row/video index mappings are mirrored.
        row_video_a_idx=ref["row_video_a_idx"],
        row_video_b_idx=ref["row_video_b_idx"],
        video_paths=ref["video_paths"],
    )
    print(f"\n  wrote {out}")
    print(f"  file size: {out.stat().st_size / 1e6:.2f} MB")

    # ---- Spot-check: re-encode 5 random instructions and compare to cache ----
    print(f"\n  spot-check: re-encode 5 random instructions and compare to cache…")
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(instructions), size=5, replace=False)
    with torch.no_grad():
        toks = tokenizer([instructions[i] for i in sample_idx]).to(args.device)
        recheck = model.encode_text(toks).float().cpu().numpy()
    max_diff_per_row = []
    for k, i in enumerate(sample_idx):
        diff = np.abs(recheck[k] - text_features[i]).max()
        max_diff_per_row.append(diff)
        print(f"    idx={i:4d}  inst={instructions[i]!r}  max_abs_diff={diff:.3e}")
    overall = max(max_diff_per_row)
    print(f"  overall max abs diff: {overall:.3e}  (pass if < 1e-4)")
    if overall >= 1e-4:
        sys.exit(f"SPOT CHECK FAILED: {overall:.3e} >= 1e-4")
    print(f"  ✅ spot check PASS")


if __name__ == "__main__":
    main()
