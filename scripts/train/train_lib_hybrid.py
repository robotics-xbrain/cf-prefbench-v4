"""Train LIBHybridSystem (Phase 3 Path B).

Same data pipeline as scripts/train_lib_robust.py, but the model
is LIBHybridSystem with a learned per-attribute gate that blends LIB
binding diffs with engineered-centroid pair-feature projections.

The 173-d engineered centroid pair features come from
outputs/auto/v3_features.npz (already produced by the original
sprint). They are standardized with mean/std fit on train rows.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
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
from cf_pref_learning.eval.metrics import aggregate_metrics, preference_flip_accuracy
from cf_pref_learning.models.lib_hybrid import LIBHybridSystem, gate_entropy_loss
from cf_pref_learning.utils.io import ensure_dir, read_jsonl, write_json, write_jsonl
from scripts.run_e2_train import (
    EVAL_SPLITS, SPLITS, _prediction_rows, _select_tie_threshold, _set_seed
)
from scripts.train_lib import _load_data, _parse_attributes
from scripts.train_lib_robust import _compute_val_pfa


def _load_centroid_features(root: Path):
    """Load the 173-d engineered centroid pair feature per row.

    Returns a dict {example_id: feat_173d}.
    """
    data = np.load(root / "outputs/auto/v3_features.npz", allow_pickle=True)
    feats = data["features"]  # [N_pair, 173]
    ids = list(data["example_ids"])
    return {str(k): feats[i].astype(np.float32) for i, k in enumerate(ids)}


def _standardize_centroid(records, centroid_dict):
    """Compute mean/std from train rows; standardize all centroid feats."""
    train_feats = np.stack(
        [centroid_dict[str(r["row"]["example_id"])] for r in records],
        axis=0,
    )
    mean = train_feats.mean(axis=0)
    std = train_feats.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def _build_indexed_tensors_hybrid(
    records, video_patches_t, centroid_dict, cent_mean, cent_std, dev, batch_select=None
):
    if batch_select is not None:
        records = [records[i] for i in batch_select]
    va_idx = torch.tensor([r["va_idx"] for r in records], dtype=torch.long, device=dev)
    vb_idx = torch.tensor([r["vb_idx"] for r in records], dtype=torch.long, device=dev)
    patches_a = video_patches_t[va_idx]
    patches_b = video_patches_t[vb_idx]
    text = torch.stack([torch.from_numpy(r["text_feat"]) for r in records], dim=0).to(dev)
    centroid_pair = np.stack(
        [centroid_dict[str(r["row"]["example_id"])] for r in records], axis=0
    )
    centroid_pair = (centroid_pair - cent_mean) / cent_std
    centroid_pair_t = torch.from_numpy(centroid_pair).to(dev)
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
    return patches_a, patches_b, centroid_pair_t, text, y, attrs


def _evaluate_hybrid(
    system, records, video_patches_t, centroid_dict, cent_mean, cent_std, dev,
    eval_batch=32, gate_override=None,
):
    system.eval()
    probs = []
    gates = []
    with torch.no_grad():
        for s in range(0, len(records), eval_batch):
            chunk = records[s:s + eval_batch]
            pa, pb, cp, txt, _, _ = _build_indexed_tensors_hybrid(
                chunk, video_patches_t, centroid_dict, cent_mean, cent_std, dev
            )
            out = system(pa, pb, cp, txt, gate_override=gate_override)
            probs.append(torch.sigmoid(out["score"]).cpu().numpy())
            gates.append(out["gate"].cpu().numpy())
    return np.concatenate(probs, axis=0), np.concatenate(gates, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/path/to/project")
    parser.add_argument("--feature-path", default="outputs/auto/v3_features_clip_patches.npz")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", default="outputs/phase3/hybrid")
    parser.add_argument("--variant", default="lib_hybrid")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr-lib", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=5e-4)
    parser.add_argument("--lr-gate", type=float, default=1e-4)
    parser.add_argument("--lr-centroid", type=float, default=5e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--d-attr", type=int, default=128)
    parser.add_argument("--n-attr", type=int, default=4)
    parser.add_argument("--lambda-bce", type=float, default=1.0)
    parser.add_argument("--lambda-recon", type=float, default=0.1)
    parser.add_argument("--lambda-cf", type=float, default=0.05)
    parser.add_argument("--lambda-para", type=float, default=0.02)
    parser.add_argument("--lambda-gate-entropy", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--ema-window", type=int, default=5)
    parser.add_argument("--ensemble-k", type=int, default=5)
    parser.add_argument("--sanity-only", action="store_true",
                        help="run gate=0 and gate=1 sanity probes and exit")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    _set_seed(args.seed)
    root = Path(args.project_root)
    out_dir = root / args.output_dir
    ensure_dir(out_dir)
    feat = Path(args.feature_path)
    if not feat.is_absolute():
        feat = root / feat
    print(f"loading patch features from {feat}", flush=True)
    rows_by_split, video_patches = _load_data(root, feat)
    train_records = [r for r in rows_by_split["train"] if r["row"]["preferred"] in {"A", "B"}]
    val_records = [r for r in rows_by_split["val"] if r["row"]["preferred"] in {"A", "B"}]
    val_all_records = rows_by_split["val"]
    print(f"loading centroid features from outputs/auto/v3_features.npz", flush=True)
    centroid_dict = _load_centroid_features(root)
    cent_mean, cent_std = _standardize_centroid(train_records, centroid_dict)
    print(f"centroid pair-feature dim: {len(cent_mean)}", flush=True)

    dev = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    video_patches_t = torch.from_numpy(video_patches).to(dev)

    patch_dim = video_patches.shape[-1]
    system = LIBHybridSystem(
        clip_patch_dim=patch_dim,
        centroid_dim=len(cent_mean),
        d_attr=args.d_attr,
        n_attr=args.n_attr,
        dropout=args.dropout,
    ).to(dev)

    if args.sanity_only:
        # Probe gate=1 (should approx Phase 1 LIB) and gate=0 (centroid-only proj)
        print("=== sanity probe: untrained ===", flush=True)
        for go in (None, 1.0, 0.0):
            p, g = _evaluate_hybrid(
                system, val_records, video_patches_t, centroid_dict, cent_mean, cent_std, dev,
                gate_override=go,
            )
            v_pfa = _compute_val_pfa(val_records, p, threshold=0.0)
            print(f"  gate_override={go}: val_PFA={v_pfa:.4f}  gate_mean={g.mean():.4f}", flush=True)
        return

    opt = torch.optim.AdamW(
        [
            {"params": system.lib.parameters(), "lr": args.lr_lib},
            {"params": system.centroid_proj.parameters(), "lr": args.lr_centroid},
            {"params": system.gate_net.parameters(), "lr": args.lr_gate},
            {"params": list(system.text_proj.parameters()) + list(system.head.parameters()),
             "lr": args.lr_head},
        ],
        weight_decay=args.wd,
    )

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
    print(f"n_train={n}  cf_pairs={len(cf_pairs)}  para_groups={len(para_groups)}",
          flush=True)
    log: list[dict[str, Any]] = []
    val_pfa_history = []
    best_smoothed = -1.0
    best_smoothed_epoch = 0
    patience_counter = 0
    checkpoint_buffer: list[tuple[int, float, dict]] = []
    early_stop_at = None
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        system.train()
        perm = np.random.permutation(n)
        run_bce = run_recon = run_cf = run_para = run_gentropy = 0.0
        steps = 0
        for s in range(0, n, args.batch):
            sel = perm[s:s + args.batch].tolist()
            pa, pb, cp, txt, y, attrs = _build_indexed_tensors_hybrid(
                train_records, video_patches_t, centroid_dict,
                cent_mean, cent_std, dev, batch_select=sel,
            )
            out = system(pa, pb, cp, txt)
            bce = F.binary_cross_entropy_with_logits(out["score"], y)
            # Reconstruction loss on the LIB-A side
            recon_loss = 0.0
            lib_recon = out["out_a"]["recon_logits"]
            for axis, logits in lib_recon.items():
                recon_loss = recon_loss + F.cross_entropy(logits, attrs[axis])
            recon_loss = recon_loss / max(1, len(lib_recon))

            cf_loss = torch.tensor(0.0, device=dev)
            if cf_pairs and args.lambda_cf > 0:
                k = min(64, len(cf_pairs))
                sel_cf = np.random.choice(len(cf_pairs), size=k, replace=False)
                ia_list, ib_list = zip(*[cf_pairs[i] for i in sel_cf])
                pa1, pb1, cp1, txt1, _, _ = _build_indexed_tensors_hybrid(
                    train_records, video_patches_t, centroid_dict,
                    cent_mean, cent_std, dev, batch_select=list(ia_list),
                )
                pa2, pb2, cp2, txt2, _, _ = _build_indexed_tensors_hybrid(
                    train_records, video_patches_t, centroid_dict,
                    cent_mean, cent_std, dev, batch_select=list(ib_list),
                )
                la = system(pa1, pb1, cp1, txt1)["score"]
                lb = system(pa2, pb2, cp2, txt2)["score"]
                cf_loss = F.binary_cross_entropy_with_logits(
                    la - lb, torch.ones_like(la)
                )

            para_loss = torch.tensor(0.0, device=dev)
            if para_groups and args.lambda_para > 0:
                gi = np.random.choice(len(para_groups),
                                     size=min(8, len(para_groups)), replace=False)
                stab = 0.0; cnt = 0
                for g_id in gi:
                    g = para_groups[g_id]
                    if len(g) < 2:
                        continue
                    pa3, pb3, cp3, txt3, _, _ = _build_indexed_tensors_hybrid(
                        train_records, video_patches_t, centroid_dict,
                        cent_mean, cent_std, dev, batch_select=g,
                    )
                    scores_g = system(pa3, pb3, cp3, txt3)["score"]
                    stab = stab + scores_g.var(unbiased=False)
                    cnt += 1
                if cnt > 0:
                    para_loss = stab / cnt

            # Gate entropy regularizer (penalize collapse to 0/1)
            gent = gate_entropy_loss(out["gate"])

            loss = (args.lambda_bce * bce
                    + args.lambda_recon * recon_loss
                    + args.lambda_cf * cf_loss
                    + args.lambda_para * para_loss
                    + args.lambda_gate_entropy * gent)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(system.parameters(), 1.0)
            opt.step()
            run_bce += float(bce.item())
            run_recon += float(recon_loss.item())
            run_cf += float(cf_loss.item()) if isinstance(cf_loss, torch.Tensor) else 0.0
            run_para += float(para_loss.item()) if isinstance(para_loss, torch.Tensor) else 0.0
            run_gentropy += float(gent.item())
            steps += 1

        # Val eval
        val_probs, val_gates = _evaluate_hybrid(
            system, val_records, video_patches_t, centroid_dict,
            cent_mean, cent_std, dev,
        )
        val_labels = np.array([1.0 if r["row"]["preferred"] == "A" else 0.0 for r in val_records])
        val_acc = float(((val_probs >= 0.5).astype(np.float32) == val_labels).mean())
        val_loss = float(-np.mean(
            val_labels * np.log(val_probs + 1e-8)
            + (1 - val_labels) * np.log(1 - val_probs + 1e-8)
        ))
        val_pfa = _compute_val_pfa(val_records, val_probs, threshold=0.0)
        val_pfa_history.append(val_pfa)
        smoothed = float(np.mean(val_pfa_history[-args.ema_window:]))

        cpu_state = {k: v.detach().cpu().clone() for k, v in system.state_dict().items()}
        checkpoint_buffer.append((epoch, smoothed, cpu_state))
        if len(checkpoint_buffer) > args.ensemble_k:
            checkpoint_buffer.pop(0)
        if smoothed > best_smoothed + 1e-6:
            best_smoothed = smoothed
            best_smoothed_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        log.append({
            "epoch": epoch,
            "train_bce": run_bce / max(1, steps),
            "train_recon": run_recon / max(1, steps),
            "train_cf": run_cf / max(1, steps),
            "train_gate_entropy": run_gentropy / max(1, steps),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_pfa": val_pfa,
            "smoothed_val_pfa": smoothed,
            "best_smoothed": best_smoothed,
            "patience_counter": patience_counter,
            "val_gate_mean": float(val_gates.mean()),
            "val_gate_per_axis_mean": [float(x) for x in val_gates.mean(axis=0)],
        })

        if epoch % 5 == 0 or epoch == 1:
            print(f"  ep{epoch:3d}  bce={log[-1]['train_bce']:.4f}  "
                  f"val_acc={val_acc:.4f}  val_pfa={val_pfa:.4f}  "
                  f"smooth_pfa={smoothed:.4f}  "
                  f"gate_mean={log[-1]['val_gate_mean']:.3f}  "
                  f"gate_per_attr={log[-1]['val_gate_per_axis_mean']}  "
                  f"patience={patience_counter}/{args.patience}  "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

        if patience_counter >= args.patience:
            early_stop_at = epoch
            print(f"  early stop at epoch {epoch}", flush=True)
            break

    # Ensemble eval
    print(f"\nEnsemble eval over {len(checkpoint_buffer)} checkpoints", flush=True)
    all_eval_records = []
    for s in EVAL_SPLITS:
        all_eval_records.extend(rows_by_split[s])

    def ensemble_eval(records):
        all_probs = []
        for ep, _, st in checkpoint_buffer:
            system.load_state_dict(st)
            p, _ = _evaluate_hybrid(
                system, records, video_patches_t, centroid_dict, cent_mean, cent_std, dev,
            )
            all_probs.append(p)
        return np.mean(np.stack(all_probs, axis=0), axis=0)

    eval_probs = ensemble_eval(all_eval_records)
    val_all_probs = ensemble_eval(val_all_records)
    threshold = _select_tie_threshold(
        [r["row"] for r in val_all_records], val_all_probs
    )
    pred_rows = _prediction_rows(
        args.variant + "_ensemble",
        [r["row"] for r in all_eval_records],
        eval_probs,
        threshold,
    )
    raw_dir = out_dir / "raw"
    ensure_dir(raw_dir)
    write_jsonl(raw_dir / f"{args.variant}_seed{args.seed}.jsonl", pred_rows)
    ensure_dir(out_dir / "logs")
    write_jsonl(out_dir / "logs" / f"train_{args.variant}_seed{args.seed}.jsonl", log)

    metrics = aggregate_metrics(pred_rows)
    metrics["impossible_premise"] = impossible_premise_metrics(pred_rows)
    metrics["by_split"] = {}
    for sp in sorted({r["split"] for r in pred_rows}):
        sub = [r for r in pred_rows if r["split"] == sp]
        metrics["by_split"][sp] = aggregate_metrics(sub)
        metrics["by_split"][sp]["impossible_premise"] = impossible_premise_metrics(sub)
    color_axis = (metrics["by_split"].get("test_heldout_color", {})
                  .get("per_axis", {}).get("color", {}).get("PFA", {}).get("value"))
    heldout_overall = metrics["by_split"].get("test_heldout_color", {}).get("PFA", {}).get("value")

    summary = {
        "variant": args.variant,
        "seed": args.seed,
        "hyperparams": vars(args),
        "epochs_trained": log[-1]["epoch"] if log else 0,
        "early_stop_at": early_stop_at,
        "best_smoothed_val_pfa": best_smoothed,
        "best_smoothed_epoch": best_smoothed_epoch,
        "tie_threshold": threshold,
        "metrics": metrics,
        "headline": {
            "color_axis_PFA": color_axis,
            "heldout_color_overall_PFA": heldout_overall,
            "val_PFA": metrics["by_split"].get("val", {}).get("PFA", {}).get("value"),
            "final_gate_per_axis": log[-1]["val_gate_per_axis_mean"] if log else None,
        },
    }
    write_json(out_dir / f"summary_{args.variant}_seed{args.seed}.json", summary)
    print(json.dumps({
        "variant": args.variant, "seed": args.seed,
        "ensemble_color_axis_PFA": color_axis,
        "ensemble_heldout_color_overall_PFA": heldout_overall,
        "epochs_trained": log[-1]["epoch"] if log else 0,
        "final_gate_per_axis": log[-1]["val_gate_per_axis_mean"] if log else None,
    }))


if __name__ == "__main__":
    main()
