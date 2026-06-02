"""Stage B: compute cos_to_train for candidate new tokens under CLIP-B/32 and ViT-L/14.

For each (token, axis):
  1. Build the held-out JSONL by string-substituting the new token into existing same-class
     held-out rows (uses test_heldout_lexical.jsonl as the source, same as
     scripts/generate_v4_axis_gen_tests.py).
  2. Encode every instruction with the requested CLIP text encoder.
  3. Compute the token's mean text feature.
  4. Compute the training mean (axis-level) text feature from train.jsonl.
  5. Cosine = cos(token_mean, train_mean).

Outputs:
  - realdata_validation/expanded_tokens/scripts/cosines_b32.json
  - realdata_validation/expanded_tokens/scripts/cosines_l14.json
  - realdata_validation/expanded_tokens/tokens_selected.md  (markdown table)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/path/to/project")
sys.path.insert(0, str(ROOT))


# Candidate tokens to validate
CANDIDATES = {
    "size": {
        # primary
        "enormous": "BIG",
        "vast": "BIG",
        "tiny": "SMALL",
        "minute": "SMALL",
        # backup (also computed, recorded for selection)
        "massive": "BIG",
        "immense": "BIG",
        "mini": "SMALL",
        "microscopic": "SMALL",
    },
    "speed": {
        # primary
        "rapidly": "FAST",
        "swiftly": "FAST",
        "leisurely": "SLOW",
        "languidly": "SLOW",
        # backup
        "hastily": "FAST",
        "expeditiously": "FAST",
        "slothfully": "SLOW",
        "languorously": "SLOW",
        "ploddingly": "SLOW",
    },
}

# Generator class membership (from scripts/generate_v4_axis_gen_tests.py)
# These are used to *find* a word to replace in the source JSONL rows.
SIZE_BIG_MEMBERS = {"large", "big", "huge"}
SIZE_SMALL_MEMBERS = {"tiny", "small"}
SPEED_FAST_MEMBERS = {"quickly", "rapidly"}
SPEED_SLOW_MEMBERS = {"slowly", "leisurely"}


def find_word(instr: str, members: set[str]) -> str | None:
    for w in instr.split():
        if w in members:
            return w
    return None


def build_token_instructions(token: str, axis: str, sem_class: str) -> list[str]:
    """Return all instructions for `token` by substitution into test_heldout_lexical.jsonl.

    This mirrors the protocol in scripts/generate_v4_axis_gen_tests.py.
    """
    sys.path.insert(0, str(ROOT))
    from cf_pref_learning.utils.io import read_jsonl

    src = list(read_jsonl(ROOT / "data/cf_prefbench_v4/test_heldout_lexical.jsonl"))
    rows = [r for r in src if r["axis"] == axis]

    if axis == "size":
        members = SIZE_BIG_MEMBERS if sem_class == "BIG" else SIZE_SMALL_MEMBERS
    else:
        members = SPEED_FAST_MEMBERS if sem_class == "FAST" else SPEED_SLOW_MEMBERS

    out = set()
    for r in rows:
        old = find_word(r["instruction"], members)
        if old is None:
            continue
        new_instr = r["instruction"].replace(old, token, 1)
        out.add(new_instr)
    return sorted(out)


def encode_texts(instructions: list[str], variant: str, device: str = "cuda") -> np.ndarray:
    import open_clip

    if variant == "ViT-B-32":
        pretrained = "/path/to/cache/clip/ViT-B-32.pt"
    elif variant == "ViT-L-14":
        pretrained = "/path/to/cache/clip/ViT-L-14.pt"
    else:
        raise ValueError(variant)

    os.environ.setdefault("HF_HOME", "/path/to/cache/huggingface")
    model, _, _ = open_clip.create_model_and_transforms(variant, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(variant)
    model = model.to(device).eval()
    feats: list[np.ndarray] = []
    B = 128
    with torch.no_grad():
        for i in range(0, len(instructions), B):
            toks = tokenizer(instructions[i : i + B]).to(device)
            f = model.encode_text(toks).cpu().numpy().astype(np.float32)
            feats.append(f)
    del model
    torch.cuda.empty_cache()
    return np.concatenate(feats, axis=0)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float((a @ b) / (na * nb))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["ViT-B-32", "ViT-L-14"])
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    from cf_pref_learning.utils.io import read_jsonl

    # 1. Collect training instructions per axis
    train_inst_by_axis: dict[str, list[str]] = defaultdict(list)
    for r in read_jsonl(ROOT / "data/cf_prefbench_v4/train.jsonl"):
        if r["axis"] in ("size", "speed"):
            train_inst_by_axis[r["axis"]].append(r["instruction"])

    # 2. Build the candidate token → instructions map
    tok_insts: dict[tuple[str, str], list[str]] = {}
    for axis, mapping in CANDIDATES.items():
        for token, sem_class in mapping.items():
            insts = build_token_instructions(token, axis, sem_class)
            tok_insts[(axis, token)] = insts
            if len(insts) == 0:
                print(f"  WARN: 0 instructions for {axis}/{token} (class={sem_class})")

    # 3. Existing held-out tokens for sanity (should match paper cosines)
    SANITY_TOKENS = [
        ("size", "colossal"), ("size", "gigantic"),
        ("size", "miniature"), ("size", "petite"),
        ("speed", "briskly"), ("speed", "speedily"),
        ("speed", "sluggishly"), ("speed", "gradually"),
    ]
    for axis, token in SANITY_TOKENS:
        f = ROOT / f"data/cf_prefbench_v4/test_heldout_{axis}_{token}.jsonl"
        if f.exists():
            insts = sorted({r["instruction"] for r in read_jsonl(f)})
            tok_insts[(axis, token)] = insts

    # 4. Encode everything once
    all_inst: set[str] = set()
    for axis, insts in train_inst_by_axis.items():
        all_inst.update(insts)
    for _, insts in tok_insts.items():
        all_inst.update(insts)
    uniq = sorted(all_inst)
    print(f"[cosine] variant={args.variant} encoding {len(uniq)} unique instructions…", flush=True)
    feats = encode_texts(uniq, args.variant)
    inst2feat = {ins: feats[k] for k, ins in enumerate(uniq)}
    print(f"  text feature dim = {feats.shape[1]}", flush=True)

    # 5. Compute axis-mean training feature and per-token cosine
    train_mean_by_axis = {
        axis: np.mean([inst2feat[ins] for ins in insts], axis=0)
        for axis, insts in train_inst_by_axis.items()
    }

    results: dict = {"variant": args.variant, "feat_dim": int(feats.shape[1]),
                     "n_train_size": len(train_inst_by_axis["size"]),
                     "n_train_speed": len(train_inst_by_axis["speed"]),
                     "tokens": {}}
    for (axis, token), insts in tok_insts.items():
        sem_class = CANDIDATES.get(axis, {}).get(token)
        if sem_class is None:
            # sanity token
            sem_class = {
                "colossal": "BIG", "gigantic": "BIG",
                "miniature": "SMALL", "petite": "SMALL",
                "briskly": "FAST", "speedily": "FAST",
                "sluggishly": "SLOW", "gradually": "SLOW",
            }[token]
        token_mean = np.mean([inst2feat[ins] for ins in insts], axis=0)
        c = cos(token_mean, train_mean_by_axis[axis])
        results["tokens"][f"{axis}/{token}"] = {
            "axis": axis, "token": token, "sem_class": sem_class,
            "n_instructions": len(insts),
            "cos_to_train": c,
            "in_cliff_zone": 0.87 <= c <= 0.97,
            "sample_instruction": insts[0] if insts else None,
        }
        print(f"  {axis}/{token:18s} ({sem_class:5s})  n={len(insts):3d}  cos={c:.4f}  "
              f"{'in-zone' if 0.87 <= c <= 0.97 else 'OUT-of-zone'}", flush=True)

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
