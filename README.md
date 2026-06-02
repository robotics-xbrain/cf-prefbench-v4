# CF-PrefBench v4 + LIB v0 — Anonymous Artifact

**Anonymous artifact** for the paper *"When Paraphrases Break Binding: A Lexical
Cliff in Small Heads over Frozen CLIP Text Embeddings."* Submitted for double-blind
review — this repository contains **no author or institution identifying information**.

> ⚠️ Anonymous notice: do not add names, affiliations, emails, or non-anonymous
> URLs. Contact: anonymous@example.com.

## 1. Paper task summary

A small learned head ("LIB v0", a per-attribute cross-attention head) is trained
over **frozen CLIP text + visual embeddings** for a controlled compositional
**preference** task: given two rendered video trajectories `(v_A, v_B)` and an
instruction `t`, predict which video matches the instruction. The paper shows a
**lexical cliff**: accuracy collapses on held-out cue variants that sit inside a
narrow CLIP-text cosine neighbourhood `[0.87, 0.97]` to training, even though the
visual input is unchanged. Failure is localised to the **text encoder** by a
substitution probe and a 2×2 encoder factorial, and replicates across
OpenCLIP-LAION-2B and SigLIP-WebLI with encoder-family-dependent token identity.

## 2. What is included

- **CF-PrefBench v4** dataset: all 8 split files (7 binding axes + impossible_premise),
  7,749 core preference rows, plus expanded per-token size/speed test splits, and the
  v3 (4-axis) benchmark for the action-axis replication. (`data/`)
- **LIB v0** model + alternative-head variants, dataset/eval code. (`cf_pref_learning/`)
- **48 trained LIB checkpoints** (`*_best.pt`): B/32 & L/14 main (8 seeds each),
  2×2 factorial (B-L, L-B), cross-family (OC-OC, SG-SG), alternative heads
  (bilinear/MLP/cross-attn/linear/CLIPScore), substitution probe, sanity. (`checkpoints/`)
- **Probe & diagnostic scripts**: substitution probe, 2×2 factorial, cross-family,
  sanity battery (SC-1…SC-6), zero-shot VLM judges, figure/table makers. (`scripts/`)
- **Released numerical results** behind every table/figure (per-seed JSON/CSV,
  `MASTER_PAPER_DATA.tex`). (`results/`)
- **Figures** (camera-ready PDFs/PNGs + generators) and the **final paper PDF**. (`figures/`, `docs/`)
- **Provenance docs**: PDF item index, evidence map, repo classification, dataset/model cards,
  reproduction guide, identity-leak scan, manifest. (`docs/`, `artifact_manifest.*`)

## 3. What is NOT included (and why)

- **Third-party pretrained encoder weights** (OpenAI CLIP, OpenCLIP-LAION-2B,
  SigLIP-WebLI): download via `open_clip` / `clip` at run time (see requirements).
- **Multi-GB frozen-CLIP feature caches** (831 MB B/32, 3.9 GB L/14, 2.3 GB SigLIP,
  ~1.7 GB OOD): excluded for size — regenerate with `scripts/eval/extract_*`.
- **Full raw videos** (62 MB): only 6 sample clips ship in `data/samples/`; regenerate
  the full set with `scripts/generate_v4_new_axes.py`.
- **Out-of-domain (Appendix M, CLEVRER/ManiSkill) per-trial predictions & eval scripts**:
  not recoverable as discrete files; reported numbers are preserved in the PDF and
  `results/MASTER_PAPER_DATA.tex`. See `docs/REPRODUCTION_GUIDE.md` §Limitations.
- Planning notes, weekly reports, prompts, review strategy, logs, VCS metadata.

## 4. Dataset overview

7 binding axes (color, object, action, spatial, size, motion_sequence, speed) + an
`impossible_premise` diagnostic split. Deterministic rendered simulator, 24-frame
trajectories at 192×144. Core total **7,749** preference rows; per-split sizes match
Table 9 (train 528/axis, val & held-out-lexical/color/spatial 84/axis, seen/camera/
hard-neg 72/axis, impossible_premise 27/non-train split). See `docs/DATASET_CARD.md`.

## 5. Model & checkpoint overview

LIB v0 = 4 attribute queries projected from the CLIP text embedding, cross-attending
over frozen CLIP patch tokens; cosine to a per-attribute "expected" projection gives a
4-d binding score, concatenated with a 16-d text projection into a 64-hidden MLP.
~0.6M trainable params (B/32) / ~0.9M (L/14). 48 `*_best.pt` checkpoints ship;
see `docs/MODEL_CARD.md` and `docs/CHECKPOINTS.md` (checkpoint→table/figure→seed map).

## 6. Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) smoke test (real integrity checks + a dry-run LIB forward)
bash scripts/smoke_test.sh

# 2) regenerate main tables 1-5 from released results (no GPU, no retraining)
bash scripts/reproduce_main_tables.sh        # -> results/reproduced_tables/

# 3) regenerate main figures 2-4 from released results
bash scripts/reproduce_figures.sh            # -> figures/reproduced/

# 4) structural verification
bash scripts/verify_artifact.sh              # -> verify_artifact_results.json
```

`scripts/*.sh` use `python3` by default; override with `PYTHON=/path/to/python`.

## 7. Reproduce main tables / figures / smoke test

- **From released results** (fast, no GPU): tables 1–5 and figures 2–4 are
  recomputed from `results/` — see steps above. Values match the PDF (Table 1, 2, 4
  exactly; Table 3 cells B-B 0.633 / B-L 0.827 / L-B 0.611 / L-L 0.803).
- **From checkpoints** (needs feature caches): regenerate caches with
  `scripts/eval/extract_*`, then run `scripts/eval/evaluate_lib_on_new_tokens.py`.
- **Full** (needs data + encoders + training): see `docs/REPRODUCTION_GUIDE.md`.

## 8. Expected hardware / runtime / disk

- Smoke test & table/figure reproduction: CPU, < 1 min, < 1 GB RAM.
- Feature extraction: 1 GPU (≥ 8 GB), minutes–hours depending on backbone.
- LIB training: 1 GPU, ~0.6–0.9M params, 60 epochs ≈ minutes per seed.
- Artifact on disk ≈ 200 MB; full feature caches if regenerated ≈ 10+ GB.

## 9. External dependencies / third-party encoders

OpenAI CLIP ViT-B/32 & ViT-L/14, OpenCLIP `ViT-B-32 laion2b_s34b_b79k`,
`ViT-B-16-SigLIP` (WebLI) — instantiated via `open_clip`/`clip`; weights are **not**
redistributed here. GPT-4o / Qwen2.5-VL judges (Appendix L) require external API/model
access and are optional.

## 10. Citation / contact / license

- Citation: *(anonymous; placeholder until de-anonymized)*.
- Contact: anonymous@example.com
- Usage: research-only; see `LICENSE_OR_USAGE.md`.

## 11. Artifact limitations

Labels are simulator-derived (no human annotations); the cliff is established on
ViT-B/32 and partially closed on ViT-L/14; the SigLIP cell varies several factors at
once (holistic family swap); Appendix M OOD per-trial artifacts are not included
(numbers preserved in the PDF). See `docs/REPRODUCTION_GUIDE.md` and the paper's
Limitations section.
