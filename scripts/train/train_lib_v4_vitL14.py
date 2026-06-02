"""Train LIB v0 on ViT-L/14 features for v4 new axes.

Almost identical to scripts.train_lib_v4_new_axes, but constructs
LIBModule and LIBPreferenceHead with clip_text_dim=768 and
clip_patch_dim=1024 to match ViT-L/14.
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
from scripts.run_e2_train import _prediction_rows, _select_tie_threshold, _set_seed
from scripts.train_lib import _build_indexed_tensors, _check_monotonic, _evaluate
from scripts.train_lib_v4_new_axes import (
    NEW_AXES, SPLITS, EVAL_SPLITS, _parse_attributes_v4,
)


class LIBSystemVitL14(nn.Module):
    """LIBSystem variant with ViT-L/14 dimensions wired through."""

    def __init__(self, n_attr=4, d_attr=128, dropout=0.3, head_hidden=64,
                  clip_text_dim=768, clip_patch_dim=1024):
        super().__init__()
        self.lib = LIBModule(
            d_attr=d_attr, n_attr=n_attr, dropout=dropout,
            clip_text_dim=clip_text_dim, clip_patch_dim=clip_patch_dim,
        )
        self.head = LIBPreferenceHead(
            n_attr=n_attr, hidden=head_hidden,
            clip_text_dim=clip_text_dim,
        )

    def forward(self, patches_a, patches_b, text):
        out_a = self.lib(patches_a, text)
        out_b = self.lib(patches_b, text)
        score = self.head(out_a["binding"], out_b["binding"], text)
        return {"score": score, "out_a": out_a, "out_b": out_b}


def _load_data(root: Path, feature_path: Path):
    data = np.load(feature_path, allow_pickle=True)
    video_patches = data["video_patches"].astype(np.float32)
    text_features = data["text_features"]
    row_va = data["row_video_a_idx"]
    row_vb = data["row_video_b_idx"]
    row_keys = list(data["example_ids"])
    key_to_idx = {k: i for i, k in enumerate(row_keys)}

    all_rows = []
    for s in SPLITS:
        for r in read_jsonl(root / "data" / "cf_prefbench_v4" / f"{s}.jsonl"):
            if r["axis"] in NEW_AXES:
                all_rows.append(r)

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
            "attrs": _parse_attributes_v4(r["instruction"]),
        }
        rows_by_split[r["split"]].append(rec)
    return rows_by_split, video_patches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/path/to/project")
    parser.add_argument("--feature-path",
                        default="outputs/auto/v4_new_axes_features_vitL14_patches.npz")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", default="outputs/phase5/exp3_vitL14")
    parser.add_argument("--variant", default="lib_v0_vitL14")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args()

    _set_seed(args.seed)
    root = Path(args.project_root)
    out_dir = root / args.output_dir
    ensure_dir(out_dir)
    feat = root / args.feature_path

    rows_by_split, video_patches = _load_data(root, feat)
    train_records = [r for r in rows_by_split["train"]
                     if r["row"]["preferred"] in {"A", "B"}]
    val_records = [r for r in rows_by_split["val"]
                   if r["row"]["preferred"] in {"A", "B"}]
    val_all_records = rows_by_split["val"]
    eval_records_by_split = {s: rows_by_split[s] for s in EVAL_SPLITS}
    print(f"train={len(train_records)}  val={len(val_records)}  "
          f"patches={video_patches.shape}  text_dim={train_records[0]['text_feat'].shape[0]}", flush=True)

    dev = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    video_patches_t = torch.from_numpy(video_patches).to(dev)

    text_dim = train_records[0]["text_feat"].shape[0]
    patch_dim = video_patches.shape[3]
    system = LIBSystemVitL14(
        d_attr=128, dropout=0.3, n_attr=4,
        clip_text_dim=text_dim, clip_patch_dim=patch_dim,
    ).to(dev)

    opt = torch.optim.AdamW(
        [{"params": system.lib.parameters(), "lr": 1e-4},
         {"params": system.head.parameters(), "lr": 5e-4}],
        weight_decay=1e-4,
    )

    by_flip = collections.defaultdict(list)
    for i, r in enumerate(train_records):
        if r["row"].get("counterfactual_flip_id"):
            by_flip[str(r["row"]["counterfactual_flip_id"])].append(i)
    cf_pairs = []
    for idxs in by_flip.values():
        a = [i for i in idxs if train_records[i]["row"]["preferred"] == "A"]
        b = [i for i in idxs if train_records[i]["row"]["preferred"] == "B"]
        for ai in a:
            for bi in b:
                cf_pairs.append((ai, bi))

    by_para = collections.defaultdict(list)
    for i, r in enumerate(train_records):
        pg = r["row"].get("paraphrase_group_id")
        if pg:
            by_para[str(pg)].append(i)
    para_groups = [v for v in by_para.values() if len(v) >= 2]

    n = len(train_records)
    best = {"val_acc": -1.0, "epoch": 0, "val_loss": float("inf"), "state": None}
    log = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        system.train()
        perm = np.random.permutation(n)
        run_bce = 0.0
        steps = 0
        for s in range(0, n, args.batch):
            sel = perm[s:s + args.batch].tolist()
            pa, pb, txt, y, attrs = _build_indexed_tensors(
                train_records, video_patches_t, dev, batch_select=sel)
            out = system(pa, pb, txt)
            bce = F.binary_cross_entropy_with_logits(out["score"], y)
            recon_loss = 0.0
            for axis, logits in out["out_a"]["recon_logits"].items():
                recon_loss = recon_loss + F.cross_entropy(logits, attrs[axis])
            recon_loss = recon_loss / 4.0
            cf_loss = torch.tensor(0.0, device=dev)
            if cf_pairs:
                k = min(32, len(cf_pairs))
                sel_cf = np.random.choice(len(cf_pairs), size=k, replace=False)
                ia, ib = zip(*[cf_pairs[i] for i in sel_cf])
                pa1, pb1, txt1, _, _ = _build_indexed_tensors(
                    train_records, video_patches_t, dev, batch_select=list(ia))
                pa2, pb2, txt2, _, _ = _build_indexed_tensors(
                    train_records, video_patches_t, dev, batch_select=list(ib))
                la = system(pa1, pb1, txt1)["score"]
                lb = system(pa2, pb2, txt2)["score"]
                cf_loss = F.binary_cross_entropy_with_logits(la - lb, torch.ones_like(la))
            para_loss = torch.tensor(0.0, device=dev)
            if para_groups:
                gi = np.random.choice(len(para_groups), size=min(8, len(para_groups)), replace=False)
                stab = 0.0; cnt = 0
                for g_id in gi:
                    g = para_groups[g_id]
                    if len(g) < 2: continue
                    pa3, pb3, txt3, _, _ = _build_indexed_tensors(
                        train_records, video_patches_t, dev, batch_select=g)
                    scores_g = system(pa3, pb3, txt3)["score"]
                    stab = stab + scores_g.var(unbiased=False); cnt += 1
                if cnt > 0:
                    para_loss = stab / cnt
            loss = 1.0 * bce + 0.1 * recon_loss + 0.05 * cf_loss + 0.02 * para_loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(system.parameters(), 1.0); opt.step()
            run_bce += float(bce.item()); steps += 1
        system.eval()
        val_probs = _evaluate(system, val_records, video_patches_t, dev)
        val_labels = np.array([1.0 if r["row"]["preferred"] == "A" else 0.0 for r in val_records])
        val_acc = float(((val_probs >= 0.5).astype(np.float32) == val_labels).mean())
        val_loss = float(-np.mean(
            val_labels * np.log(val_probs + 1e-8)
            + (1 - val_labels) * np.log(1 - val_probs + 1e-8)
        ))
        log.append({"epoch": epoch, "train_bce": run_bce/max(steps,1), "val_acc": val_acc, "val_loss": val_loss})
        if val_acc > best["val_acc"]:
            best = {"val_acc": val_acc, "epoch": epoch, "val_loss": val_loss,
                    "state": {k: v.detach().cpu().clone() for k, v in system.state_dict().items()}}
        if epoch % 10 == 0 or epoch == 1:
            print(f"  ep{epoch:3d} bce={log[-1]['train_bce']:.4f}  val={val_acc:.4f}  elapsed={time.time()-t0:.1f}s", flush=True)

    system.load_state_dict(best["state"])
    system.eval()

    if args.save_checkpoint:
        ckpt_path = out_dir / f"lib_v0_vitL14_seed{args.seed}_best.pt"
        torch.save(best["state"], ckpt_path)
        print(f"saved checkpoint {ckpt_path}", flush=True)

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
    summary = {
        "variant": args.variant, "seed": args.seed,
        "best_epoch": best["epoch"], "best_val_acc": best["val_acc"],
        "best_val_loss": best["val_loss"], "tie_threshold": threshold,
        "metrics": metrics,
        "hyperparams": vars(args),
    }
    write_json(out_dir / f"summary_{args.variant}_seed{args.seed}.json", summary)
    ensure_dir(out_dir / "logs")
    write_jsonl(out_dir / "logs" / f"train_{args.variant}_seed{args.seed}.jsonl", log)
    print(json.dumps({"variant": args.variant, "seed": args.seed,
                       "best_epoch": best["epoch"], "val_acc": best["val_acc"]}))


if __name__ == "__main__":
    main()
