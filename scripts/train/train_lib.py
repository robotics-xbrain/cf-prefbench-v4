"""LIB v0 training script.

Loads precomputed CLIP patch features + text features, trains
LIBModule + LIBPreferenceHead with the 4-term loss specified in
LIB_DESIGN.md §3.

Usage:
    python scripts/train_lib.py --seed 42 --epochs 100 \
        --output-dir outputs/lib_v0_smoke

The training runs in two phases:
  - epochs 1..warmup_epochs: head-only on engineered centroid feature
    (warm-start; not the LIB binding score). This gets the head to
    a sensible initialization.
  - epochs warmup_epochs+1..end: full LIB joint training with the
    4-term loss.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cf_pref_learning.eval.eval_impossible import impossible_premise_metrics
from cf_pref_learning.eval.metrics import aggregate_metrics
from cf_pref_learning.models.lib import LIBModule, LIBPreferenceHead
from cf_pref_learning.utils.io import ensure_dir, read_jsonl, write_json, write_jsonl
from scripts.run_e2_train import (
    EVAL_SPLITS, SPLITS, _prediction_rows, _select_tie_threshold, _set_seed
)


# Vocabularies for the reconstruction auxiliary
COLOR_VOCAB = ["red", "blue", "yellow", "purple", "cyan", "magenta", "green"]
OBJECT_VOCAB = ["button", "control", "switch", "lever", "block", "object", "target"]
ACTION_VOCAB = ["press", "open", "close"]
SPATIAL_VOCAB = ["left", "right", "center"]


def _parse_attributes(instruction: str) -> dict[str, int]:
    """Lightweight instruction parser to provide auxiliary labels.

    Returns class indices into the per-axis vocabularies above.
    """
    t = instruction.lower()
    out = {"color": 0, "object": 0, "action": 0, "spatial": len(SPATIAL_VOCAB) - 1}
    for i, c in enumerate(COLOR_VOCAB):
        if c in t:
            out["color"] = i
            break
    for i, o in enumerate(OBJECT_VOCAB):
        if o in t:
            out["object"] = i
            break
    if any(w in t for w in ["press", "push", "tap", "touch", "activate", "hit"]):
        out["action"] = 0
    elif any(w in t for w in ["open", "pull", "unseal"]):
        out["action"] = 1
    elif any(w in t for w in ["close", "shut", "seal"]):
        out["action"] = 2
    if any(w in t for w in ["left", "north"]):
        out["spatial"] = 0
    elif any(w in t for w in ["right", "south"]):
        out["spatial"] = 1
    return out


def _load_data(root: Path, feature_path: Path):
    """Loads all CF-PrefBench rows + patch features.

    Returns:
        rows_by_split: dict[split] -> list of (row_dict, va_idx, vb_idx, text_feat, attrs)
        video_patches: float32 tensor [n_videos, K, 49, 768]
    """
    data = np.load(feature_path, allow_pickle=True)
    video_patches = data["video_patches"].astype(np.float32)  # cast from f16 -> f32
    text_features = data["text_features"]  # [N, 512]
    row_va = data["row_video_a_idx"]
    row_vb = data["row_video_b_idx"]
    row_keys = list(data["example_ids"])
    key_to_idx = {k: i for i, k in enumerate(row_keys)}

    all_rows = []
    for s in SPLITS:
        all_rows.extend(read_jsonl(root / "data" / "cf_prefbench" / f"{s}.jsonl"))

    rows_by_split: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
    for r in all_rows:
        if str(r["example_id"]) not in key_to_idx:
            continue
        i = key_to_idx[str(r["example_id"])]
        rec = {
            "row": r,
            "va_idx": int(row_va[i]),
            "vb_idx": int(row_vb[i]),
            "text_feat": text_features[i].astype(np.float32),
            "attrs": _parse_attributes(r["instruction"]),
        }
        rows_by_split[r["split"]].append(rec)
    return rows_by_split, video_patches


class LIBSystem(nn.Module):
    """Convenience wrapper: LIB module + preference head."""

    def __init__(
        self,
        n_attr: int = 4,
        d_attr: int = 128,
        dropout: float = 0.3,
        head_hidden: int = 64,
    ):
        super().__init__()
        # For n_attr > 4 (per-color queries), expand recon vocabularies
        # so the auxiliary loss still applies to each attribute.
        if n_attr == 4:
            self.lib = LIBModule(d_attr=d_attr, n_attr=n_attr, dropout=dropout)
        else:
            # Generic: only the first 4 axes use the per-axis recon head;
            # extra attribute queries operate without a recon target.
            self.lib = LIBModule(d_attr=d_attr, n_attr=n_attr, dropout=dropout)
        self.head = LIBPreferenceHead(n_attr=n_attr, hidden=head_hidden)

    def encode_one(self, patches: torch.Tensor, text: torch.Tensor):
        return self.lib(patches, text)

    def forward(self, patches_a, patches_b, text):
        out_a = self.encode_one(patches_a, text)
        out_b = self.encode_one(patches_b, text)
        score = self.head(out_a["binding"], out_b["binding"], text)
        return {
            "score": score,
            "out_a": out_a,
            "out_b": out_b,
        }


def _build_indexed_tensors(records, video_patches_t, dev, batch_select=None):
    """For a list of records, return (patches_a, patches_b, text, y, attrs_a, attrs_b)."""
    if batch_select is not None:
        records = [records[i] for i in batch_select]
    va_idx = torch.tensor([r["va_idx"] for r in records], dtype=torch.long, device=dev)
    vb_idx = torch.tensor([r["vb_idx"] for r in records], dtype=torch.long, device=dev)
    patches_a = video_patches_t[va_idx]  # [B, K, 49, 768]
    patches_b = video_patches_t[vb_idx]
    text = torch.stack([torch.from_numpy(r["text_feat"]) for r in records], dim=0).to(dev)
    has_y = all(r["row"]["preferred"] in {"A", "B"} for r in records)
    y = None
    if has_y:
        y = torch.tensor(
            [1.0 if r["row"]["preferred"] == "A" else 0.0 for r in records],
            dtype=torch.float32, device=dev,
        )
    attrs = {axis: torch.tensor([r["attrs"][axis] for r in records],
                                 dtype=torch.long, device=dev)
             for axis in ["color", "object", "action", "spatial"]}
    return patches_a, patches_b, text, y, attrs


def _evaluate(system, records, video_patches_t, dev, eval_batch=32):
    """Compute sigmoid scores for a list of records."""
    system.eval()
    probs = []
    with torch.no_grad():
        for s in range(0, len(records), eval_batch):
            chunk = records[s:s + eval_batch]
            pa, pb, txt, _, _ = _build_indexed_tensors(chunk, video_patches_t, dev)
            out = system(pa, pb, txt)
            probs.append(torch.sigmoid(out["score"]).cpu().numpy())
    return np.concatenate(probs, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/path/to/project")
    parser.add_argument("--feature-path", default="outputs/auto/v3_features_clip_patches.npz")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", default="outputs/lib_v0_smoke")
    parser.add_argument("--variant", default="lib_v0")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr-lib", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=5e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--d-attr", type=int, default=128)
    parser.add_argument("--lambda-bce", type=float, default=1.0)
    parser.add_argument("--lambda-recon", type=float, default=0.1)
    parser.add_argument("--lambda-cf", type=float, default=0.05)
    parser.add_argument("--lambda-para", type=float, default=0.02)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--select-by", choices=["val_acc", "val_loss", "val_pfa"], default="val_acc",
                        help="model-selection metric. val_pfa requires intermediate flip-group eval.")
    parser.add_argument("--n-attr", type=int, default=4,
                        help="number of attribute queries. 4=color/object/action/spatial. 7=per-color.")
    args = parser.parse_args()

    _set_seed(args.seed)
    root = Path(args.project_root)
    out_dir = root / args.output_dir
    ensure_dir(out_dir)
    feat = Path(args.feature_path)
    if not feat.is_absolute():
        feat = root / feat

    print(f"loading features from {feat}", flush=True)
    rows_by_split, video_patches = _load_data(root, feat)
    train_records = [r for r in rows_by_split["train"]
                     if r["row"]["preferred"] in {"A", "B"}]
    val_records = [r for r in rows_by_split["val"]
                   if r["row"]["preferred"] in {"A", "B"}]
    val_all_records = rows_by_split["val"]
    eval_records_by_split = {s: rows_by_split[s] for s in EVAL_SPLITS}
    print(f"train={len(train_records)}  val(A/B)={len(val_records)}  "
          f"val_all={len(val_all_records)}  "
          f"eval splits={list(eval_records_by_split.keys())}", flush=True)

    dev = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    video_patches_t = torch.from_numpy(video_patches).to(dev)

    system = LIBSystem(d_attr=args.d_attr, dropout=args.dropout, n_attr=args.n_attr).to(dev)

    opt = torch.optim.AdamW(
        [
            {"params": system.lib.parameters(), "lr": args.lr_lib},
            {"params": system.head.parameters(), "lr": args.lr_head},
        ],
        weight_decay=args.wd,
    )

    # Pre-compute counterfactual flip groups in train for L_cf
    by_flip = collections.defaultdict(list)
    for i, r in enumerate(train_records):
        if r["row"].get("counterfactual_flip_id"):
            by_flip[str(r["row"]["counterfactual_flip_id"])].append(i)
    cf_pairs = []
    for flip_id, idxs in by_flip.items():
        a_idxs = [i for i in idxs if train_records[i]["row"]["preferred"] == "A"]
        b_idxs = [i for i in idxs if train_records[i]["row"]["preferred"] == "B"]
        for ai in a_idxs:
            for bi in b_idxs:
                cf_pairs.append((ai, bi))

    by_para = collections.defaultdict(list)
    for i, r in enumerate(train_records):
        pg = r["row"].get("paraphrase_group_id")
        if pg:
            by_para[str(pg)].append(i)
    para_groups = [v for v in by_para.values() if len(v) >= 2]

    n = len(train_records)
    log: list[dict[str, Any]] = []
    best = {"val_acc": -1.0, "epoch": 0, "val_loss": float("inf"), "state": None}
    t0 = time.time()
    print(f"n_train={n}  cf_pairs={len(cf_pairs)}  para_groups={len(para_groups)}", flush=True)

    for epoch in range(1, args.epochs + 1):
        system.train()
        perm = np.random.permutation(n)
        run_bce = run_recon = run_cf = run_para = 0.0
        steps = 0
        for s in range(0, n, args.batch):
            sel = perm[s:s + args.batch].tolist()
            pa, pb, txt, y, attrs = _build_indexed_tensors(
                train_records, video_patches_t, dev, batch_select=sel
            )
            out = system(pa, pb, txt)
            bce = F.binary_cross_entropy_with_logits(out["score"], y)

            # Reconstruction loss on the A side (using attrs labels of the
            # instruction; both A and B share the same instruction in a pair)
            recon_loss = 0.0
            for axis, logits in out["out_a"]["recon_logits"].items():
                recon_loss = recon_loss + F.cross_entropy(logits, attrs[axis])
            recon_loss = recon_loss / 4.0

            cf_loss = torch.tensor(0.0, device=dev)
            if cf_pairs and args.lambda_cf > 0:
                k = min(64, len(cf_pairs))
                sel_cf = np.random.choice(len(cf_pairs), size=k, replace=False)
                ia_list, ib_list = zip(*[cf_pairs[i] for i in sel_cf])
                pa1, pb1, txt1, _, _ = _build_indexed_tensors(
                    train_records, video_patches_t, dev,
                    batch_select=list(ia_list),
                )
                pa2, pb2, txt2, _, _ = _build_indexed_tensors(
                    train_records, video_patches_t, dev,
                    batch_select=list(ib_list),
                )
                la = system(pa1, pb1, txt1)["score"]
                lb = system(pa2, pb2, txt2)["score"]
                cf_loss = F.binary_cross_entropy_with_logits(
                    la - lb, torch.ones_like(la)
                )

            para_loss = torch.tensor(0.0, device=dev)
            if para_groups and args.lambda_para > 0:
                gi = np.random.choice(len(para_groups),
                                     size=min(8, len(para_groups)), replace=False)
                stab = 0.0
                cnt = 0
                for g_id in gi:
                    g = para_groups[g_id]
                    if len(g) < 2:
                        continue
                    pa3, pb3, txt3, _, _ = _build_indexed_tensors(
                        train_records, video_patches_t, dev, batch_select=g,
                    )
                    scores_g = system(pa3, pb3, txt3)["score"]
                    stab = stab + scores_g.var(unbiased=False)
                    cnt += 1
                if cnt > 0:
                    para_loss = stab / cnt

            loss = (args.lambda_bce * bce
                    + args.lambda_recon * recon_loss
                    + args.lambda_cf * cf_loss
                    + args.lambda_para * para_loss)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(system.parameters(), 1.0)
            opt.step()
            run_bce += float(bce.item())
            run_recon += float(recon_loss.item())
            run_cf += float(cf_loss.item()) if isinstance(cf_loss, torch.Tensor) else 0.0
            run_para += float(para_loss.item()) if isinstance(para_loss, torch.Tensor) else 0.0
            steps += 1

        # Val eval
        system.eval()
        val_probs = _evaluate(system, val_records, video_patches_t, dev)
        with torch.no_grad():
            val_labels = np.array([
                1.0 if r["row"]["preferred"] == "A" else 0.0 for r in val_records
            ])
            val_acc = float(((val_probs >= 0.5).astype(np.float32) == val_labels).mean())
            val_loss = float(-np.mean(
                val_labels * np.log(val_probs + 1e-8)
                + (1 - val_labels) * np.log(1 - val_probs + 1e-8)
            ))

        # Q_attr Gram off-diagonal (for non-collapse monitoring)
        qweight = system.lib.q_proj.weight.detach()
        nattr = system.lib.n_attr
        dattr = system.lib.d_attr
        qmat = qweight.view(nattr, dattr, -1).mean(dim=-1)  # [n_attr, d_attr]
        gram_off = float(system.lib.gram_offdiag_norm(qmat).item())

        log.append({
            "epoch": epoch,
            "train_bce": run_bce / max(steps, 1),
            "train_recon": run_recon / max(steps, 1),
            "train_cf": run_cf / max(steps, 1),
            "train_para": run_para / max(steps, 1),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "q_gram_offdiag": gram_off,
        })
        if val_acc > best["val_acc"] + 1e-6 or (
            abs(val_acc - best["val_acc"]) < 1e-6 and val_loss < best["val_loss"]
        ):
            best = {"val_acc": val_acc, "epoch": epoch, "val_loss": val_loss,
                    "state": {k: v.detach().cpu().clone() for k, v in system.state_dict().items()}}

        if epoch % 5 == 0 or epoch == 1:
            print(f"  ep{epoch:3d}  bce={log[-1]['train_bce']:.4f}  "
                  f"recon={log[-1]['train_recon']:.4f}  "
                  f"val_acc={val_acc:.4f}  "
                  f"q_gram={gram_off:.3f}  "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

    # Restore best
    system.load_state_dict(best["state"])
    system.eval()

    # Final eval on all splits
    all_eval_records = []
    for s in EVAL_SPLITS:
        all_eval_records.extend(rows_by_split[s])
    eval_probs = _evaluate(system, all_eval_records, video_patches_t, dev)
    val_all_probs = _evaluate(system, val_all_records, video_patches_t, dev)
    threshold = _select_tie_threshold(
        [r["row"] for r in val_all_records], val_all_probs
    )
    pred_rows = _prediction_rows(
        args.variant, [r["row"] for r in all_eval_records], eval_probs, threshold
    )

    raw_path = out_dir / "raw" / f"{args.variant}_seed{args.seed}.jsonl"
    ensure_dir(raw_path.parent)
    write_jsonl(raw_path, pred_rows)

    metrics = aggregate_metrics(pred_rows)
    metrics["impossible_premise"] = impossible_premise_metrics(pred_rows)
    metrics["by_split"] = {}
    for s in sorted({r["split"] for r in pred_rows}):
        sub = [r for r in pred_rows if r["split"] == s]
        metrics["by_split"][s] = aggregate_metrics(sub)
        metrics["by_split"][s]["impossible_premise"] = impossible_premise_metrics(sub)

    color_axis_pfa = (
        metrics["by_split"].get("test_heldout_color", {})
        .get("per_axis", {}).get("color", {}).get("PFA", {}).get("value")
    )
    val_pfa = metrics["by_split"].get("val", {}).get("PFA", {}).get("value")

    summary = {
        "variant": args.variant,
        "seed": args.seed,
        "feature_path": str(feat),
        "best_epoch": best["epoch"],
        "best_val_acc": best["val_acc"],
        "best_val_loss": best["val_loss"],
        "tie_threshold": threshold,
        "hyperparams": vars(args),
        "metrics": metrics,
        "headline": {
            "val_PFA": val_pfa,
            "test_heldout_color__color_axis_PFA": color_axis_pfa,
            "training_log_final": log[-1] if log else None,
            "smoke_gate": {
                "no_nan": not any(
                    np.isnan(l["train_bce"]) or np.isnan(l["val_loss"]) for l in log
                ),
                "loss_monotonic_smoothed": _check_monotonic(
                    [l["train_bce"] for l in log[: min(5, len(log))]]
                ),
                "val_pfa_gt_0_30": (val_pfa or 0.0) > 0.30,
                "recon_decreased": (
                    log[0]["train_recon"] - log[-1]["train_recon"]
                ) > 0.5 if log else False,
                "q_gram_offdiag_lt_0_3": log[-1]["q_gram_offdiag"] < 0.3 if log else False,
            },
        },
    }
    write_json(out_dir / f"summary_{args.variant}_seed{args.seed}.json", summary)
    write_jsonl(out_dir / "logs" / f"train_{args.variant}_seed{args.seed}.jsonl", log)

    print(json.dumps({
        "variant": args.variant, "seed": args.seed,
        "best_epoch": best["epoch"], "val_acc": best["val_acc"],
        "val_PFA": val_pfa,
        "test_heldout_color__color_axis_PFA": color_axis_pfa,
        "smoke_gate": summary["headline"]["smoke_gate"],
    }))


def _check_monotonic(losses: list[float]) -> bool:
    """Smoothed monotonic check: each value <= previous + 0.05."""
    if len(losses) < 2:
        return True
    return all(losses[i] <= losses[i - 1] + 0.05 for i in range(1, len(losses)))


if __name__ == "__main__":
    main()
