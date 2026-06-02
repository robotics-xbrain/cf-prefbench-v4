"""Stage C: evaluate the 8 *new* held-out tokens on an LIB-v0 checkpoint.

Mirror of experiments/EXP-B/eval_crossenc_on_cliff_tokens.py, but with the new-token
registry only. The cell argument selects the text encoder / patch features (B-B or L-L).

Outputs: realdata_validation/expanded_tokens/predictions/<basename>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/path/to/project")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/EXP-B"))

from cf_pref_learning.utils.io import read_jsonl, write_json  # noqa: E402
from scripts.train_lib import LIBSystem  # noqa: E402
from scripts.train_lib_v4_vitL14 import LIBSystemVitL14  # noqa: E402


# 8 new held-out tokens
NEW_TOKENS = [
    # size BIG, SMALL
    ("enormous", "size", "test_heldout_size_enormous.jsonl"),
    ("vast",     "size", "test_heldout_size_vast.jsonl"),
    ("tiny",     "size", "test_heldout_size_tiny.jsonl"),
    ("minute",   "size", "test_heldout_size_minute.jsonl"),
    # speed FAST, SLOW
    ("swiftly",   "speed", "test_heldout_speed_swiftly.jsonl"),
    ("hastily",   "speed", "test_heldout_speed_hastily.jsonl"),
    ("leisurely", "speed", "test_heldout_speed_leisurely.jsonl"),
    ("languidly", "speed", "test_heldout_speed_languidly.jsonl"),
]

CLASS_GROUPS = {
    "size_BIG":   ["enormous", "vast"],
    "size_SMALL": ["tiny", "minute"],
    "speed_FAST": ["swiftly", "hastily"],
    "speed_SLOW": ["leisurely", "languidly"],
}

SEM_CLASS = {
    "enormous": "BIG", "vast": "BIG",
    "tiny": "SMALL", "minute": "SMALL",
    "swiftly": "FAST", "hastily": "FAST",
    "leisurely": "SLOW", "languidly": "SLOW",
}


def cell_text_variant(cell: str) -> str:
    return {"B-B": "ViT-B-32", "L-L": "ViT-L-14"}[cell]


def cell_patch_npz(cell: str) -> str:
    return {"B-B": "outputs/auto/v4_new_axes_features_clip_patches.npz",
            "L-L": "outputs/auto/v4_new_axes_features_vitL14_patches.npz"}[cell]


def encode_texts(instructions: list[str], variant: str, device: str = "cuda") -> np.ndarray:
    import open_clip
    pretrained = {"ViT-B-32": "/path/to/cache/clip/ViT-B-32.pt",
                  "ViT-L-14": "/path/to/cache/clip/ViT-L-14.pt"}[variant]
    os.environ.setdefault("HF_HOME", "/path/to/cache/huggingface")
    model, _, _ = open_clip.create_model_and_transforms(variant, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(variant)
    model = model.to(device).eval()
    out: list[np.ndarray] = []
    B = 128
    with torch.no_grad():
        for i in range(0, len(instructions), B):
            toks = tokenizer(instructions[i:i + B]).to(device)
            f = model.encode_text(toks).cpu().numpy().astype(np.float32)
            out.append(f)
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out, axis=0)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64); b = b.astype(np.float64)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float((a @ b) / (na * nb))


def build_system(cell: str, n_attr=4, d_attr=128, dropout=0.3):
    if cell == "B-B":
        return LIBSystem(d_attr=d_attr, dropout=dropout, n_attr=n_attr)
    elif cell == "L-L":
        return LIBSystemVitL14(d_attr=d_attr, dropout=dropout, n_attr=n_attr,
                               clip_text_dim=768, clip_patch_dim=1024)
    raise ValueError(cell)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cell", choices=["B-B", "L-L"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    t0 = time.time()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    text_variant = cell_text_variant(args.cell)
    patch_npz = ROOT / cell_patch_npz(args.cell)

    print(f"[new-token eval] cell={args.cell} text={text_variant} "
          f"patch={patch_npz.name}", flush=True)
    print(f"  checkpoint: {ckpt_path}", flush=True)

    pdata = np.load(patch_npz, allow_pickle=True)
    video_patches = pdata["video_patches"].astype(np.float32)
    video_paths = list(pdata["video_paths"])
    v2i = {v: i for i, v in enumerate(video_paths)}

    # Training instructions per axis
    train_inst_by_axis = defaultdict(list)
    for r in read_jsonl(ROOT / "data/cf_prefbench_v4/train.jsonl"):
        if r["axis"] in ("size", "speed"):
            train_inst_by_axis[r["axis"]].append(r["instruction"])

    rows_by_token: dict[str, list] = {}
    instructions_by_token: dict[str, list] = {}
    for token, axis, fname in NEW_TOKENS:
        rows = list(read_jsonl(ROOT / f"data/cf_prefbench_v4/{fname}"))
        rows = [r for r in rows if r["axis"] == axis and
                r["video_a"] in v2i and r["video_b"] in v2i]
        rows_by_token[token] = rows
        instructions_by_token[token] = sorted({r["instruction"] for r in rows})

    n_rows_total = sum(len(rs) for rs in rows_by_token.values())
    print(f"  total new-token rows: {n_rows_total} across {len(rows_by_token)} tokens",
          flush=True)

    # Encode all needed text
    all_inst: list[str] = []
    for token in instructions_by_token:
        all_inst.extend(instructions_by_token[token])
    for axis_insts in train_inst_by_axis.values():
        all_inst.extend(axis_insts)
    uniq = sorted(set(all_inst))
    print(f"  encoding {len(uniq)} unique instructions with {text_variant}…", flush=True)
    feats = encode_texts(uniq, text_variant)
    inst2feat = {ins: feats[k] for k, ins in enumerate(uniq)}

    train_mean_by_axis = {
        axis: np.mean([inst2feat[i] for i in insts], axis=0)
        for axis, insts in train_inst_by_axis.items()
    }

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system = build_system(args.cell).to(dev)
    state = torch.load(ckpt_path, map_location="cpu")
    system.load_state_dict(state, strict=True)
    system.eval()
    print(f"  system loaded "
          f"({sum(p.numel() for p in system.parameters() if p.requires_grad):,} params)",
          flush=True)

    video_patches_t = torch.from_numpy(video_patches).to(dev)

    per_token: dict[str, dict] = {}
    per_row_preds: list[dict] = []  # for CSV dump downstream
    for token, axis, _fname in NEW_TOKENS:
        rows = rows_by_token[token]
        scores: list[np.ndarray] = []
        B = 64
        with torch.no_grad():
            for s in range(0, len(rows), B):
                chunk = rows[s:s + B]
                va = torch.tensor([v2i[r["video_a"]] for r in chunk], device=dev)
                vb = torch.tensor([v2i[r["video_b"]] for r in chunk], device=dev)
                pa = video_patches_t[va]; pb = video_patches_t[vb]
                txt = torch.from_numpy(
                    np.stack([inst2feat[r["instruction"]] for r in chunk]).astype(np.float32)
                ).to(dev)
                out = system(pa, pb, txt)
                scores.append(torch.sigmoid(out["score"]).cpu().numpy())
        scores = np.concatenate(scores)
        preds = ["A" if x >= 0.5 else "B" for x in scores]
        correct = sum(1 for p, r in zip(preds, rows) if p == r["preferred"])
        acc = correct / max(1, len(rows))
        token_mean = np.mean([inst2feat[i] for i in instructions_by_token[token]], axis=0)
        c = cos(token_mean, train_mean_by_axis[axis])
        per_token[token] = {
            "axis": axis, "sem_class": SEM_CLASS[token],
            "n_rows": len(rows), "correct": correct,
            "accuracy": acc, "cos_to_train": c,
        }
        for r, p, sc in zip(rows, preds, scores):
            per_row_preds.append({
                "token": token, "axis": axis, "sem_class": SEM_CLASS[token],
                "example_id": r["example_id"], "instruction": r["instruction"],
                "video_a": r["video_a"], "video_b": r["video_b"],
                "pred": p, "score": float(sc), "preferred": r["preferred"],
                "correct": int(p == r["preferred"]),
            })
        print(f"  {token:10s} axis={axis:5s}  n={len(rows):3d}  acc={acc:.4f}  cos={c:.4f}",
              flush=True)

    class_aggs = {}
    for cls, tokens in CLASS_GROUPS.items():
        accs = [per_token[t]["accuracy"] for t in tokens]
        cs = [per_token[t]["cos_to_train"] for t in tokens]
        class_aggs[cls] = {
            "tokens": tokens,
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs, ddof=0)),
            "cos_to_train_mean": float(np.mean(cs)),
            "n_rows_total": sum(per_token[t]["n_rows"] for t in tokens),
        }

    overall = float(np.mean([per_token[t]["accuracy"] for t in per_token]))

    result = {
        "cell": args.cell, "seed": args.seed,
        "checkpoint": str(ckpt_path),
        "text_encoder": text_variant, "patch_npz": str(patch_npz),
        "n_rows_total": n_rows_total,
        "new_token_mean_accuracy": overall,
        "per_token": per_token,
        "class_aggregates": class_aggs,
        "elapsed_sec": time.time() - t0,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, result)

    # Per-row CSV alongside the JSON
    csv_path = out_path.with_suffix(".csv")
    import csv
    keys = ["token", "axis", "sem_class", "example_id", "instruction", "video_a",
            "video_b", "pred", "score", "preferred", "correct"]
    with csv_path.open("w") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in per_row_preds:
            writer.writerow(row)
    print(f"\n  new_token_mean_acc = {overall:.4f}", flush=True)
    print(f"  wrote {out_path}", flush=True)
    print(f"  wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
