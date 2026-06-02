# Reproduction guide

## Quickstart (CPU, < 1 minute)

```bash
pip install -r requirements.txt
bash scripts/smoke_test.sh            # integrity + dry-run LIB forward
bash scripts/reproduce_main_tables.sh # tables 1-5 -> results/reproduced_tables/
bash scripts/reproduce_figures.sh     # figures 2-4 -> figures/reproduced/
bash scripts/verify_artifact.sh       # structure check
```

## Three reproduction tiers

### Tier 1 — from released results (no GPU, no model)
Tables 1–5 and Figures 2–4 are recomputed purely from `results/` JSON/CSV by
`scripts/artifact/reproduce_main_tables.py` and `reproduce_figures.py`. This needs
only numpy (+ matplotlib for figures). Cross-check against `docs/EMNLP_final.pdf`
and `results/MASTER_PAPER_DATA.tex` (the committed single source of truth for every
number). Verified matches: Table 1, Table 2, Table 4 exact; Table 3 cells
B-B 0.633 / B-L 0.827 / L-B 0.611 / L-L 0.803.

### Tier 2 — from checkpoints (1 GPU)
1. Regenerate the frozen-CLIP feature caches (excluded for size):
   ```bash
   python scripts/eval/extract_clip_patch_features.py            # ViT-B/32
   python scripts/eval/extract_v4_new_axes_features_vitL14.py    # ViT-L/14
   python scripts/eval/extract_v4_new_axes_text_openclip_laion2b.py  # OpenCLIP-LAION-2B
   ```
   (these download the encoders via `open_clip`/`clip`.)
2. Evaluate the shipped checkpoints on the held-out token splits:
   ```bash
   python scripts/eval/evaluate_lib_on_new_tokens.py   # per-token cliff (Tables 5,7,18)
   python scripts/eval/evaluate_lib_on_scoot.py        # 4-verb probe (Table 1)
   ```
   See `docs/CHECKPOINTS.md` for the checkpoint→table/figure→seed mapping.

### Tier 3 — full (data generation + training)
```bash
python scripts/generate_v4_new_axes.py                # regenerate videos + splits
python scripts/eval/extract_clip_patch_features.py    # features
python scripts/train/train_lib.py                     # train LIB v0 (B/32)
python scripts/train/train_lib_v4_vitL14.py           # ViT-L/14
python scripts/train/train_lib_v4_with_sanity.py      # + substitution / SC probes
```

## Hardware / runtime / disk

| Task | Hardware | Runtime | Disk |
|---|---|---|---|
| smoke test, table/figure repro | CPU | < 1 min | < 1 GB |
| feature extraction (B/32) | 1 GPU ≥ 8 GB | ~minutes | 831 MB cache |
| feature extraction (L/14) | 1 GPU ≥ 16 GB | ~tens of min | 3.9 GB cache |
| LIB training (per seed) | 1 GPU | ~minutes (0.6–0.9M params, 60 epochs, batch 32) | small |
| full caches if regenerated | — | — | 10+ GB |

## Training recipe (from the paper)
AdamW, 60 epochs, 3 seeds (B/32 main also has an 8-seed pool); losses =
preference BCE + reconstruction CE + counterfactual margin + paraphrase stability
(weights 1.0 / 0.1 / 0.05 / 0.02); lr_LIB 1e-4, lr_head 5e-4, lr_preproj 1e-4,
weight-decay 1e-4, dropout 0.3. L/14 uses batch 16. Cross-family / B-L cells add a
learnable linear pre-projection to a fixed 512 before LIB.

## What reproduces from what
- **Released CSV/JSON only:** Tables 1–5, 7, 18, 19; Figures 2–6 (re-render).
- **Checkpoint eval (needs caches):** any per-token/per-seed cliff number.
- **Full data generation + training:** the dataset and checkpoints themselves.

## Known limitations
- **Appendix M (CLEVRER & ManiSkill OOD):** the per-trial prediction files and OOD
  speed-cliff eval scripts were **not recoverable** as discrete files; the oversized
  feature caches (~1.7 GB) and raw video arrays were excluded for size. The reported
  numbers (CLEVRER ≈ chance 0.49; ManiSkill B/32 FAST 0.33 / SLOW 0.62, substitution
  → 0.64; L/14 null) are preserved in `docs/EMNLP_final.pdf` and
  `results/MASTER_PAPER_DATA.tex` but cannot be re-derived from shipped files. Flagged
  as a WARNING in `FINAL_ARTIFACT_REPORT.md`.
- **Third-party encoder weights** are downloaded at run time, not shipped.
- **Figure 1** (LIB v0 architecture) is a hand-drawn diagram; authoritative rendering
  is `docs/EMNLP_final.pdf` page 3.
- The figure file → final-PDF figure-number mapping is documented in
  `figures/README.md` (camera-ready filenames use the older section numbering).

## Exact verification commands
```bash
PYTHON=python3 bash scripts/smoke_test.sh        # -> smoke_test_results.json
PYTHON=python3 bash scripts/verify_artifact.sh   # -> verify_artifact_results.json
diff <(python scripts/artifact/reproduce_main_tables.py) /dev/null  # writes reproduced tables
```
