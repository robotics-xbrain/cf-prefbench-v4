"""Retrain LIB v0 seed 1 on v4 new axes and run SC-6 (shuffle instruction
text features) + a verb-substitution sanity test at the end.

Reuses scripts.train_lib_v4_new_axes for training, then evaluates with:
  - NORMAL: as-is
  - SC-6 (shuffle): replace per-row text features with a random shuffle of
    held-out row text features
  - ZERO: text features set to zero
  - SUBSTITUTION: replace each "transit ..." instruction's text feature
    with the corresponding "move ..." text feature (same direction tokens)
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cf_pref_learning.utils.io import ensure_dir, read_jsonl, write_json
from scripts.run_e2_train import _set_seed
from scripts.train_lib import LIBSystem, _build_indexed_tensors, _evaluate
from scripts.train_lib_v4_new_axes import (
    NEW_AXES, SPLITS, _load_data,
)


def _evaluate_with_text_override(system, records, video_patches_t, dev,
                                  text_override: np.ndarray, eval_batch=32):
    """Evaluate the model with custom per-row text features (overrides records' text_feat)."""
    system.eval()
    probs = []
    with torch.no_grad():
        for s in range(0, len(records), eval_batch):
            chunk = records[s:s + eval_batch]
            va_idx = torch.tensor([r["va_idx"] for r in chunk], dtype=torch.long, device=dev)
            vb_idx = torch.tensor([r["vb_idx"] for r in chunk], dtype=torch.long, device=dev)
            pa = video_patches_t[va_idx]
            pb = video_patches_t[vb_idx]
            txt = torch.from_numpy(text_override[s:s + eval_batch].astype(np.float32)).to(dev)
            out = system(pa, pb, txt)
            probs.append(torch.sigmoid(out["score"]).cpu().numpy())
    return np.concatenate(probs, axis=0)


def _row_accuracy(records, probs, threshold=0.5):
    if not records:
        return None
    correct = 0
    n = 0
    for i, r in enumerate(records):
        if r["row"]["preferred"] not in {"A", "B"}:
            continue
        pred = "A" if probs[i] >= threshold else "B"
        if pred == r["row"]["preferred"]:
            correct += 1
        n += 1
    return correct / n if n else None


def _per_verb_accuracy(records, probs, threshold=0.5):
    out = {}
    by_verb = collections.defaultdict(list)
    for i, r in enumerate(records):
        v = r["row"]["instruction"].split()[0]
        by_verb[v].append(i)
    for v, idxs in by_verb.items():
        correct = 0; n = 0
        for i in idxs:
            r = records[i]
            if r["row"]["preferred"] not in {"A", "B"}:
                continue
            pred = "A" if probs[i] >= threshold else "B"
            if pred == r["row"]["preferred"]:
                correct += 1
            n += 1
        out[v] = correct / n if n else None
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/path/to/project")
    parser.add_argument("--feature-path",
                        default="outputs/auto/v4_new_axes_features_clip_patches.npz")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs/phase5/exp2_motion_verify/sc6")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=32)
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
    heldout_records = rows_by_split["test_heldout_lexical"]
    print(f"train={len(train_records)}  val={len(val_records)}  heldout_lex={len(heldout_records)}", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_patches_t = torch.from_numpy(video_patches).to(dev)

    system = LIBSystem(d_attr=128, dropout=0.3, n_attr=4).to(dev)
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
    best = {"val_acc": -1.0, "state": None}
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        system.train()
        perm = np.random.permutation(n)
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
                k = min(64, len(cf_pairs))
                sel_cf = np.random.choice(len(cf_pairs), size=k, replace=False)
                ia_list, ib_list = zip(*[cf_pairs[i] for i in sel_cf])
                pa1, pb1, txt1, _, _ = _build_indexed_tensors(
                    train_records, video_patches_t, dev, batch_select=list(ia_list))
                pa2, pb2, txt2, _, _ = _build_indexed_tensors(
                    train_records, video_patches_t, dev, batch_select=list(ib_list))
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
        # Quick val
        system.eval()
        val_probs = _evaluate(system, val_records, video_patches_t, dev)
        val_labels = np.array([1.0 if r["row"]["preferred"] == "A" else 0.0 for r in val_records])
        val_acc = float(((val_probs >= 0.5).astype(np.float32) == val_labels).mean())
        if val_acc > best["val_acc"]:
            best = {"val_acc": val_acc, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in system.state_dict().items()}}
        if epoch % 10 == 0 or epoch == 1:
            print(f"  ep{epoch:3d} val={val_acc:.4f}  elapsed={time.time()-t0:.1f}s", flush=True)

    system.load_state_dict(best["state"])
    system.eval()
    print(f"best val_acc {best['val_acc']:.4f} at epoch {best['epoch']}", flush=True)
    torch.save(best["state"], out_dir / f"lib_v0_seed{args.seed}_best.pt")

    # --- Sanity battery ---
    print(flush=True)
    print("=== SANITY BATTERY (heldout_lexical only) ===", flush=True)
    heldout_lex_AB = [r for r in heldout_records if r["row"]["preferred"] in {"A", "B"}]
    by_axis = collections.defaultdict(list)
    for r in heldout_lex_AB:
        by_axis[r["row"]["axis"]].append(r)

    # NORMAL
    text_feats = np.stack([r["text_feat"] for r in heldout_lex_AB])
    normal_probs = _evaluate_with_text_override(
        system, heldout_lex_AB, video_patches_t, dev, text_feats)

    # ZERO instruction text
    zero_probs = _evaluate_with_text_override(
        system, heldout_lex_AB, video_patches_t, dev, np.zeros_like(text_feats))

    # SHUFFLE instruction text features across rows (SC-6 style)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(heldout_lex_AB))
    shuffle_text = text_feats[perm]
    shuffle_probs = _evaluate_with_text_override(
        system, heldout_lex_AB, video_patches_t, dev, shuffle_text)

    # SUBSTITUTION: for each held-out row, find a train instruction with the same
    # direction tokens and replace text feat with that train instruction's text feat
    train_text = collections.defaultdict(list)  # (axis, dir_tuple) -> [text_feats]
    for r in train_records:
        instr = r["row"]["instruction"]
        # Crude key: axis + token-set of direction words
        dirs = tuple(w for w in instr.split() if w in {"left", "right", "up", "down"})
        size_words = tuple(w for w in instr.split() if w in
                            {"large", "small", "big", "tiny", "huge"})
        speed_words = tuple(w for w in instr.split() if w in
                              {"quickly", "slowly", "rapidly", "leisurely"})
        key = (r["row"]["axis"], dirs, size_words, speed_words)
        train_text[key].append(r["text_feat"])
    sub_text = []
    sub_skipped = 0
    for r in heldout_lex_AB:
        instr = r["row"]["instruction"]
        dirs = tuple(w for w in instr.split() if w in {"left", "right", "up", "down"})
        size_words = tuple(w for w in instr.split() if w in
                            {"large", "small", "big", "tiny", "huge"})
        speed_words = tuple(w for w in instr.split() if w in
                              {"quickly", "slowly", "rapidly", "leisurely"})
        key = (r["row"]["axis"], dirs, size_words, speed_words)
        if key in train_text:
            sub_text.append(train_text[key][rng.integers(len(train_text[key]))])
        else:
            sub_skipped += 1
            sub_text.append(r["text_feat"])
    sub_text = np.stack(sub_text)
    sub_probs = _evaluate_with_text_override(
        system, heldout_lex_AB, video_patches_t, dev, sub_text)
    print(f"  substitution skipped (no train match): {sub_skipped}/{len(heldout_lex_AB)}", flush=True)

    # Summarize
    results = {}
    for cond_name, probs in [("normal", normal_probs),
                              ("zero_text", zero_probs),
                              ("shuffle_text", shuffle_probs),
                              ("substitution_with_train_verb", sub_probs)]:
        print(f"--- {cond_name} ---", flush=True)
        cond_summary = {}
        for axis, recs in by_axis.items():
            idx = [i for i, r in enumerate(heldout_lex_AB) if r["row"]["axis"] == axis]
            sub_probs_axis = probs[idx]
            recs_subset = [heldout_lex_AB[i] for i in idx]
            acc = _row_accuracy(recs_subset, sub_probs_axis)
            per_verb = _per_verb_accuracy(recs_subset, sub_probs_axis)
            cond_summary[axis] = {"accuracy": acc, "per_verb": per_verb}
            verb_str = "  ".join(f"{v}={a:.3f}" for v, a in sorted(per_verb.items()))
            print(f"  {axis:18s}  acc={acc:.4f}  per_verb: {verb_str}", flush=True)
        results[cond_name] = cond_summary

    write_json(out_dir / f"sanity_v4_seed{args.seed}.json", results)
    print(f"\nwrote {out_dir / f'sanity_v4_seed{args.seed}.json'}", flush=True)


if __name__ == "__main__":
    main()
