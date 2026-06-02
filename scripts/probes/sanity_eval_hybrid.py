"""Sanity-check evaluation for LIBHybridSystem.

Trains the hybrid on seed 42 (same recipe as train_lib_hybrid.py),
then runs 6 sanity probes at end-of-training using the same ensemble
of checkpoints. Each probe perturbs ONE component of the forward
pass and re-evaluates color-axis PFA on test_heldout_color (axis=color).

The probes:
  SC-1: force gate = 1.0 (pure LIB path)
  SC-2: force gate = 0.0 (pure centroid projection path)
  SC-3: random gate ~ Uniform(0, 1) per-instance at inference
  SC-4: zero the centroid pair feature at inference
  SC-5: zero the patch features at inference (kills LIB)
  SC-6: shuffle instruction text within batch
"""

from __future__ import annotations

import argparse
import collections
import copy
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
from cf_pref_learning.eval.metrics import aggregate_metrics, preference_flip_accuracy
from cf_pref_learning.models.lib_hybrid import LIBHybridSystem, gate_entropy_loss
from cf_pref_learning.utils.io import ensure_dir, write_json, write_jsonl
from scripts.run_e2_train import (
    EVAL_SPLITS, SPLITS, _prediction_rows, _select_tie_threshold, _set_seed
)
from scripts.train_lib import _load_data, _parse_attributes
from scripts.train_lib_robust import _compute_val_pfa
from scripts.train_lib_hybrid import (
    _load_centroid_features, _standardize_centroid,
    _build_indexed_tensors_hybrid, _evaluate_hybrid,
)


def _color_axis_pfa_from_pred_rows(pred_rows):
    """Pull the color-axis PFA on test_heldout_color from a list of pred rows."""
    sub = [r for r in pred_rows
           if r["split"] == "test_heldout_color" and r.get("axis") == "color"]
    by_flip = collections.defaultdict(list)
    for r in sub:
        fid = r.get("counterfactual_flip_id")
        if fid and r.get("preferred") in {"A", "B"} and r.get("prediction") in {"A", "B"}:
            by_flip[str(fid)].append(r)
    elig = [g for g in by_flip.values() if len({x["preferred"] for x in g}) >= 2]
    if not elig:
        return float("nan"), 0
    c = sum(1 for g in elig if all(x["prediction"] == x["preferred"] for x in g))
    return c / len(elig), len(elig)


def _heldout_color_overall_from_pred_rows(pred_rows):
    """Pull heldout_color OVERALL PFA (cross-axis)."""
    sub = [r for r in pred_rows if r["split"] == "test_heldout_color"]
    by_flip = collections.defaultdict(list)
    for r in sub:
        fid = r.get("counterfactual_flip_id")
        if fid and r.get("preferred") in {"A", "B"} and r.get("prediction") in {"A", "B"}:
            by_flip[str(fid)].append(r)
    elig = [g for g in by_flip.values() if len({x["preferred"] for x in g}) >= 2]
    if not elig:
        return float("nan")
    c = sum(1 for g in elig if all(x["prediction"] == x["preferred"] for x in g))
    return c / len(elig)


def _ensemble_eval_with_perturbation(
    system, checkpoint_buffer, records, video_patches_t, centroid_dict,
    cent_mean, cent_std, dev, perturbation_fn=None, eval_batch=32,
):
    """Run ensemble eval. `perturbation_fn` is called per batch to
    perturb (patches_a, patches_b, centroid_pair, text) -> perturbed
    versions; returns (patches_a, patches_b, centroid_pair, text, gate_override).
    """
    all_probs = []
    for ep, _, st in checkpoint_buffer:
        system.load_state_dict(st)
        system.eval()
        probs = []
        for s in range(0, len(records), eval_batch):
            chunk = records[s:s + eval_batch]
            pa, pb, cp, txt, _, _ = _build_indexed_tensors_hybrid(
                chunk, video_patches_t, centroid_dict, cent_mean, cent_std, dev
            )
            gate_override = None
            if perturbation_fn is not None:
                pa, pb, cp, txt, gate_override = perturbation_fn(pa, pb, cp, txt)
            with torch.no_grad():
                out = system(pa, pb, cp, txt, gate_override=gate_override)
                probs.append(torch.sigmoid(out["score"]).cpu().numpy())
        all_probs.append(np.concatenate(probs, axis=0))
    return np.mean(np.stack(all_probs, axis=0), axis=0)


def _train_hybrid(args, root):
    """Adapted from scripts/train_lib_hybrid.main; returns:
    (system, checkpoint_buffer, val_all_records, eval_records_by_split,
     video_patches_t, centroid_dict, cent_mean, cent_std, dev, threshold_normal,
     pred_rows_normal, log)
    """
    _set_seed(args.seed)
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
    log = []
    val_pfa_history = []
    best_smoothed = -1.0
    patience_counter = 0
    checkpoint_buffer = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        system.train()
        perm = np.random.permutation(n)
        for s in range(0, n, args.batch):
            sel = perm[s:s + args.batch].tolist()
            pa, pb, cp, txt, y, attrs = _build_indexed_tensors_hybrid(
                train_records, video_patches_t, centroid_dict,
                cent_mean, cent_std, dev, batch_select=sel,
            )
            out = system(pa, pb, cp, txt)
            bce = F.binary_cross_entropy_with_logits(out["score"], y)
            recon_loss = 0.0
            for axis, logits in out["out_a"]["recon_logits"].items():
                recon_loss = recon_loss + F.cross_entropy(logits, attrs[axis])
            recon_loss = recon_loss / max(1, len(out["out_a"]["recon_logits"]))
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
                cf_loss = F.binary_cross_entropy_with_logits(la - lb, torch.ones_like(la))
            para_loss = torch.tensor(0.0, device=dev)
            if para_groups and args.lambda_para > 0:
                gi = np.random.choice(len(para_groups), size=min(8, len(para_groups)), replace=False)
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
                    stab = stab + scores_g.var(unbiased=False); cnt += 1
                if cnt > 0:
                    para_loss = stab / cnt
            gent = gate_entropy_loss(out["gate"])
            loss = (args.lambda_bce * bce + args.lambda_recon * recon_loss
                    + args.lambda_cf * cf_loss + args.lambda_para * para_loss
                    + args.lambda_gate_entropy * gent)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(system.parameters(), 1.0)
            opt.step()
        val_probs, _ = _evaluate_hybrid(
            system, val_records, video_patches_t, centroid_dict, cent_mean, cent_std, dev
        )
        val_pfa = _compute_val_pfa(val_records, val_probs, threshold=0.0)
        val_pfa_history.append(val_pfa)
        smoothed = float(np.mean(val_pfa_history[-args.ema_window:]))
        cpu_state = {k: v.detach().cpu().clone() for k, v in system.state_dict().items()}
        checkpoint_buffer.append((epoch, smoothed, cpu_state))
        if len(checkpoint_buffer) > args.ensemble_k:
            checkpoint_buffer.pop(0)
        if smoothed > best_smoothed + 1e-6:
            best_smoothed = smoothed
            patience_counter = 0
        else:
            patience_counter += 1
        log.append({"epoch": epoch, "val_pfa": val_pfa, "smoothed_val_pfa": smoothed})
        if epoch % 10 == 0 or epoch == 1:
            print(f"  ep{epoch:3d}  val_pfa={val_pfa:.4f}  smooth={smoothed:.4f}  "
                  f"patience={patience_counter}/{args.patience}  elapsed={time.time()-t0:.1f}s",
                  flush=True)
        if patience_counter >= args.patience:
            print(f"  early stop at {epoch}", flush=True)
            break

    return (system, checkpoint_buffer, val_all_records, rows_by_split,
            video_patches_t, centroid_dict, cent_mean, cent_std, dev)


def _make_pred_rows_from_probs(records, probs, threshold):
    return _prediction_rows(
        "sanity_eval", [r["row"] for r in records], probs, threshold
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/path/to/project")
    parser.add_argument("--feature-path", default="outputs/auto/v3_features_clip_patches.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/phase4/sanity")
    parser.add_argument("--epochs", type=int, default=60)
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
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = Path(args.project_root)
    out_dir = root / args.output_dir
    ensure_dir(out_dir)

    print(f"=== Train hybrid (seed {args.seed}) ===", flush=True)
    (system, checkpoint_buffer, val_all_records, rows_by_split,
     video_patches_t, centroid_dict, cent_mean, cent_std, dev) = _train_hybrid(args, root)

    print(f"\n=== Build eval set ===", flush=True)
    all_eval_records = []
    for s in EVAL_SPLITS:
        all_eval_records.extend(rows_by_split[s])

    # Pre-fit tie threshold using NORMAL (no-perturbation) ensemble
    print(f"\n=== Normal (no-perturbation) baseline ===", flush=True)
    normal_probs_val = _ensemble_eval_with_perturbation(
        system, checkpoint_buffer, val_all_records, video_patches_t,
        centroid_dict, cent_mean, cent_std, dev,
    )
    threshold = _select_tie_threshold([r["row"] for r in val_all_records], normal_probs_val)
    print(f"  selected tie threshold: {threshold}", flush=True)
    normal_probs = _ensemble_eval_with_perturbation(
        system, checkpoint_buffer, all_eval_records, video_patches_t,
        centroid_dict, cent_mean, cent_std, dev,
    )
    pred_normal = _make_pred_rows_from_probs(all_eval_records, normal_probs, threshold)
    ca_normal, n_normal = _color_axis_pfa_from_pred_rows(pred_normal)
    overall_normal = _heldout_color_overall_from_pred_rows(pred_normal)
    print(f"  NORMAL color-axis PFA = {ca_normal:.4f} (n={n_normal})", flush=True)
    print(f"  NORMAL heldout_color overall = {overall_normal:.4f}", flush=True)

    results = {
        "seed": args.seed,
        "tie_threshold": threshold,
        "normal": {"color_axis_PFA": ca_normal, "heldout_color_overall_PFA": overall_normal, "n": n_normal},
        "sanity_checks": {},
    }

    # SC-1: force gate = 1.0 (pure LIB path)
    print(f"\n=== SC-1: force gate = 1.0 (pure LIB) ===", flush=True)
    def perturb_sc1(pa, pb, cp, txt):
        return pa, pb, cp, txt, 1.0
    probs = _ensemble_eval_with_perturbation(
        system, checkpoint_buffer, all_eval_records, video_patches_t,
        centroid_dict, cent_mean, cent_std, dev, perturbation_fn=perturb_sc1,
    )
    pred = _make_pred_rows_from_probs(all_eval_records, probs, threshold)
    ca, n = _color_axis_pfa_from_pred_rows(pred); overall = _heldout_color_overall_from_pred_rows(pred)
    print(f"  SC-1 color-axis PFA = {ca:.4f}  expected ~0.57-0.64 (LIB v0 baseline)", flush=True)
    results["sanity_checks"]["SC1_gate_1"] = {
        "color_axis_PFA": ca, "heldout_color_overall_PFA": overall, "n": n,
        "expected": "approx 0.57-0.64 (LIB v0 baseline)",
    }
    write_jsonl(out_dir / "sc1_force_gate_1.jsonl", pred[:50])

    # SC-2: force gate = 0.0 (pure centroid path through centroid_proj)
    print(f"\n=== SC-2: force gate = 0.0 (pure centroid projection) ===", flush=True)
    def perturb_sc2(pa, pb, cp, txt):
        return pa, pb, cp, txt, 0.0
    probs = _ensemble_eval_with_perturbation(
        system, checkpoint_buffer, all_eval_records, video_patches_t,
        centroid_dict, cent_mean, cent_std, dev, perturbation_fn=perturb_sc2,
    )
    pred = _make_pred_rows_from_probs(all_eval_records, probs, threshold)
    ca, n = _color_axis_pfa_from_pred_rows(pred); overall = _heldout_color_overall_from_pred_rows(pred)
    print(f"  SC-2 color-axis PFA = {ca:.4f}  expected ~0.929 (engineered centroid)", flush=True)
    results["sanity_checks"]["SC2_gate_0"] = {
        "color_axis_PFA": ca, "heldout_color_overall_PFA": overall, "n": n,
        "expected": "approx 0.929 (engineered centroid)",
    }
    write_jsonl(out_dir / "sc2_force_gate_0.jsonl", pred[:50])

    # SC-3: random gate per-instance at inference (no learned routing)
    print(f"\n=== SC-3: random gate at inference ===", flush=True)
    rng = np.random.default_rng(args.seed)
    def perturb_sc3(pa, pb, cp, txt):
        # We can't use gate_override directly because we want per-instance random.
        # Strategy: override gate by hacking the gate_net's bias for this batch.
        # Simpler: set gate_override as None, but pre-randomize the gate by adding
        # noise to the text. That's not clean.
        # Cleanest: use a callback hook. But we have gate_override that broadcasts a scalar.
        # We'll fall back to: scalar gate override sampled per call (so per-batch random).
        # NOTE: this is approximate; per-instance random would require model edit.
        return pa, pb, cp, txt, float(rng.uniform(0.0, 1.0))
    # Multiple draws and average over them
    rng = np.random.default_rng(args.seed)
    probs_avg = None
    for k_draw in range(10):
        probs = _ensemble_eval_with_perturbation(
            system, checkpoint_buffer, all_eval_records, video_patches_t,
            centroid_dict, cent_mean, cent_std, dev, perturbation_fn=perturb_sc3,
        )
        probs_avg = probs if probs_avg is None else probs_avg + probs
    probs_avg /= 10
    pred = _make_pred_rows_from_probs(all_eval_records, probs_avg, threshold)
    ca, n = _color_axis_pfa_from_pred_rows(pred); overall = _heldout_color_overall_from_pred_rows(pred)
    print(f"  SC-3 color-axis PFA = {ca:.4f}  (10 random-gate draws averaged)", flush=True)
    print(f"    expected: in between 0.643 (LIB) and 0.929 (centroid)", flush=True)
    results["sanity_checks"]["SC3_random_gate"] = {
        "color_axis_PFA": ca, "heldout_color_overall_PFA": overall, "n": n,
        "expected": "in [0.64, 0.93] (between LIB and centroid)",
    }

    # SC-4: zero centroid pair feature
    print(f"\n=== SC-4: zero centroid pair feature ===", flush=True)
    def perturb_sc4(pa, pb, cp, txt):
        return pa, pb, torch.zeros_like(cp), txt, None
    probs = _ensemble_eval_with_perturbation(
        system, checkpoint_buffer, all_eval_records, video_patches_t,
        centroid_dict, cent_mean, cent_std, dev, perturbation_fn=perturb_sc4,
    )
    pred = _make_pred_rows_from_probs(all_eval_records, probs, threshold)
    ca, n = _color_axis_pfa_from_pred_rows(pred); overall = _heldout_color_overall_from_pred_rows(pred)
    print(f"  SC-4 color-axis PFA = {ca:.4f}  (centroid zeroed, learned gate stays)", flush=True)
    print(f"    expected: significantly lower than 0.9048 if centroid carries the signal", flush=True)
    results["sanity_checks"]["SC4_centroid_zeroed"] = {
        "color_axis_PFA": ca, "heldout_color_overall_PFA": overall, "n": n,
        "expected": "much lower than 0.9048 if centroid carries the signal",
    }

    # SC-5: zero patch features (kills LIB)
    print(f"\n=== SC-5: zero patch features (LIB blind) ===", flush=True)
    def perturb_sc5(pa, pb, cp, txt):
        return torch.zeros_like(pa), torch.zeros_like(pb), cp, txt, None
    probs = _ensemble_eval_with_perturbation(
        system, checkpoint_buffer, all_eval_records, video_patches_t,
        centroid_dict, cent_mean, cent_std, dev, perturbation_fn=perturb_sc5,
    )
    pred = _make_pred_rows_from_probs(all_eval_records, probs, threshold)
    ca, n = _color_axis_pfa_from_pred_rows(pred); overall = _heldout_color_overall_from_pred_rows(pred)
    print(f"  SC-5 color-axis PFA = {ca:.4f}  (LIB blind, centroid + learned gate)", flush=True)
    print(f"    expected: approx 0.929 if centroid carries the signal", flush=True)
    results["sanity_checks"]["SC5_lib_zeroed"] = {
        "color_axis_PFA": ca, "heldout_color_overall_PFA": overall, "n": n,
        "expected": "approx 0.929 (centroid carries the signal)",
    }

    # SC-6: shuffle instruction text within batch
    print(f"\n=== SC-6: shuffle text within batch ===", flush=True)
    def perturb_sc6(pa, pb, cp, txt):
        perm = torch.randperm(txt.shape[0], device=txt.device)
        return pa, pb, cp, txt[perm], None
    # Average over several shuffles for stability
    probs_avg = None
    for k_draw in range(10):
        probs = _ensemble_eval_with_perturbation(
            system, checkpoint_buffer, all_eval_records, video_patches_t,
            centroid_dict, cent_mean, cent_std, dev, perturbation_fn=perturb_sc6,
        )
        probs_avg = probs if probs_avg is None else probs_avg + probs
    probs_avg /= 10
    pred = _make_pred_rows_from_probs(all_eval_records, probs_avg, threshold)
    ca, n = _color_axis_pfa_from_pred_rows(pred); overall = _heldout_color_overall_from_pred_rows(pred)
    print(f"  SC-6 color-axis PFA = {ca:.4f}  (text shuffled, instruction broken)", flush=True)
    print(f"    expected: significantly lower if model uses instruction", flush=True)
    results["sanity_checks"]["SC6_shuffle_text"] = {
        "color_axis_PFA": ca, "heldout_color_overall_PFA": overall, "n": n,
        "expected": "significantly lower than 0.9048 if instruction is used",
    }

    write_json(out_dir / "sanity_summary.json", results)
    print(f"\n=== Summary ===", flush=True)
    print(json.dumps({k: (v.get("color_axis_PFA") if isinstance(v, dict) else v)
                      for k, v in {**{"NORMAL": results["normal"]},
                                    **results["sanity_checks"]}.items()}, indent=2))


if __name__ == "__main__":
    main()
