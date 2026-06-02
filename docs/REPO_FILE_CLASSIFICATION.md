# REPO FILE CLASSIFICATION — CF-PrefBench v4 anonymous artifact

Scope decisions follow `PAPER_ITEM_INDEX.md` / `PAPER_EVIDENCE_MAP.md`. Labels:
`INCLUDE_MAIN`, `INCLUDE_APPENDIX`, `INCLUDE_OPTIONAL`, `INCLUDE_OPTIONAL_CANDIDATE`,
`EXCLUDE_PRIVATE`, `EXCLUDE_UNRELATED`, `EXCLUDE_IDENTITY_RISK`, `EXCLUDE_TOO_LARGE`,
`EXCLUDE_DUPLICATE`, `EXCLUDE_OLD_DRAFT`. Identity-risk column flags presence of
absolute paths / author strings to be scrubbed on copy (text only).

## Code / package

| Path | Type | Size | Matched item | Label | Reason | Identity-risk | Notes |
|---|---|---|---|---|---|---|---|
| `cf_pref_learning/` | pkg | 572K | LIB v0, data, eval | INCLUDE_MAIN | core model+data+eval package | low (docstrings) | scrub paths |
| `cf_pref_learning/models/lib*.py` | py | — | Fig 1, Fig 6, App H | INCLUDE_MAIN | LIB v0 + alt heads | low | |
| `configs/` | json | 32K | E0–E6 configs | INCLUDE_OPTIONAL | run configs | low | |
| `scripts/train_lib*.py`, `extract_*`, `run_gpt4o*/qwen2vl*`, `make_*`, `build_*table*`, `evaluate_lib_*` | py | subset of 1.8M | Tables/Figs | INCLUDE_MAIN/APPENDIX | training/eval/probe/figure/table scripts | medium (paths) | scrub paths |
| `scripts/` (metaworld gen, autopilot, job-queue, weekly orchestration) | py | — | — | EXCLUDE_UNRELATED | not tied to final PDF items | — | |
| `make_cross_family*.py`, `make_expanded_cliff*.py` (top-level) | py | ~33K | Fig 3, Fig 4 | INCLUDE_APPENDIX | figure generators (keep latest + note) | low | |
| `_tools/` | dir | 47M | — | EXCLUDE_UNRELATED | dev tooling, not paper | — | |

## Data

| Path | Type | Size | Matched item | Label | Reason | Identity-risk | Notes |
|---|---|---|---|---|---|---|---|
| `data/cf_prefbench_v4/` | jsonl | 6.2M | Tables 5,7,9,10; all v4 splits | INCLUDE_MAIN | the released benchmark | low | |
| `data/cf_prefbench/` (v3) | jsonl | 3.3M | Tables 12,13 (App G) | INCLUDE_APPENDIX | v3 action-axis replication | low | |
| `data/raw/v4_new_axes/generation_summary.json` | json | 4K | Tables 9,10 | INCLUDE_APPENDIX | split/pool provenance | low | |
| `data/raw/v3_mujoco_scripted/generation_summary.json` | json | 4K | App G | INCLUDE_APPENDIX | v3 provenance | low | |
| `data/raw/*/videos/` | mp4 | 62M | benchmark videos | EXCLUDE_TOO_LARGE | ship ~6 sample mp4 only; regenerate via scripts | low | small sample → `data/samples/` |
| `data/raw/metaworld*/` | mixed | ~30M | — | EXCLUDE_UNRELATED | legacy source not in final v4 | — | |

## Checkpoints (best.pt only; epoch_*.pt excluded)

| Path | Type | Size | Matched item | Label | Reason |
|---|---|---|---|---|---|
| `experiments/EXP-B/B-B_published/seed{1,2,3}/*_best.pt` | pt | ~12.6M | T1,T2,T3,T6,T11 | INCLUDE_MAIN | B/32 main |
| `outputs/phase5/exp3b_vitL14/checkpoints/lib_v0_vitL14_seed{1,2,3}_best.pt` | pt | ~12M | T2,T3,T5 | INCLUDE_MAIN | L/14 main |
| `experiments/EXP-F/B-32/seed{7,31,99,256,2025}/*_best.pt` | pt | ~13.5M | T5,T7 (n=8 pool) | INCLUDE_APPENDIX | B/32 extra seeds |
| `experiments/EXP-F/L-14/seed{7,31,99,256,2025}/*_best.pt` | pt | ~19.5M | T5,T7 (n=8 pool) | INCLUDE_APPENDIX | L/14 extra seeds |
| `experiments/EXP-B/{B-L,L-B}/seed*/*_best.pt` | pt | ~28M | T3, Fig2 | INCLUDE_MAIN | factorial cells |
| `experiments/EXP-H/{OC-OC,SG-SG}/seed{42,123,2024}/*_best.pt` | pt | ~24M | T4,T18,Fig3 | INCLUDE_MAIN | cross-family |
| `experiments/EXP-H/B-OC/...` | pt | ~11M | (not in final PDF §4.4) | INCLUDE_OPTIONAL_CANDIDATE | misaligned cell; not reported → optional |
| `experiments/EXP-D/{bilinear,mlp,crossattn}/seed*/ckpt_best.pt` | pt | ~12M | Fig6,T8 | INCLUDE_APPENDIX | alt heads |
| `experiments/EXP-D/{linear,clipscore}/seed*/ckpt_best.pt` | pt | ~6.5M | Fig6 omitted baselines | INCLUDE_OPTIONAL | "degenerate to chance" baselines |
| `experiments/EXP-G/seed{42,123,2024}/*_best.pt` | pt | ~8M | T1,T6 | INCLUDE_APPENDIX | substitution-probe seeds |
| `outputs/phase5/exp2_motion_verify/sc6/lib_v0_seed1_best.pt` | pt | 2.7M | T16 | INCLUDE_APPENDIX | sanity case study 2 |
| `experiments/EXP-*/seed*/ckpt_epoch_*.pt` | pt | ~400M | — | EXCLUDE_TOO_LARGE | intermediate epochs; best.pt suffices |

## Result files

| Path | Type | Size | Matched item | Label | Reason |
|---|---|---|---|---|---|
| `outputs/phase5/MASTER_PAPER_DATA.tex` | tex | 32K | ALL | INCLUDE_MAIN | single source of truth for numbers |
| `outputs/phase5/exp3_aggregate/cliff_data_B32.json` | json | 3K | T1,T6 | INCLUDE_MAIN | |
| `outputs/phase5/exp3b_vitL14/L14_cliff_table.json` | json | 2K | T2,Fig5 | INCLUDE_MAIN | |
| `outputs/phase5/exp4_axis_gen/cliff_table.json` (+CROSS_AXIS_VERDICT.md) | json/md | 11K | T3,T4,T5,T7,T18 | INCLUDE_MAIN | |
| `outputs/phase5/exp5_vlm/{gpt4o,qwen}/*.json` + predictions | json/jsonl | 736K | T19 | INCLUDE_APPENDIX | VLM decodability |
| `outputs/phase5/exp1_axes/REPORT.md` + jsons | mixed | 2.3M | T11,T12,T13 | INCLUDE_APPENDIX | 7-axis + v3 action |
| `outputs/phase5/exp2_motion_verify/*.json` | json | (jsons only) | T16,T17,App J | INCLUDE_APPENDIX | sanity raw data |
| `outputs/phase5/exp6_cross_sanity/*` | json | 164K | T15 | INCLUDE_APPENDIX | hybrid sanity |
| `outputs/phase5/section*_paper/*.md` | md | ~800K | narrative provenance | INCLUDE_OPTIONAL | scrub paths |
| `experiments/EXP-{B,D,F,G,H}/**/cliff_tokens_seed*.json` | json | ~150K | T1,T3,T4,T5,T8,T18,Fig6 | INCLUDE_MAIN/APPENDIX | per-seed raw cliff metrics |
| `realdata_validation/expanded_tokens/{tables,predictions,scripts,*.md}` | mixed | 1.5M | T5,T7,App B,C | INCLUDE_APPENDIX | n=8 expanded-token pool |
| `outputs/all_results.json`, `outputs/auto/*`, `outputs/e2_main/*`, `outputs/track_a*/*` | mixed | GBs | superseded/dev | EXCLUDE_UNRELATED / EXCLUDE_DUPLICATE | not tied to final-PDF items |

## Feature caches

| Path | Size | Label | Reason |
|---|---|---|---|
| `outputs/auto/_smoke_siglip_test.npz` | 19M | INCLUDE_OPTIONAL | purpose-built smoke-test cache (video_patches+text_features) |
| `outputs/auto/v4_new_axes_features_openclip_b32_laion2b.npz` | 6.5M | INCLUDE_OPTIONAL | OC text features (cross-family) |
| `outputs/auto/v3_features_clip_patches.npz` (831M), `*_vitL14_patches.npz` (3.9G), `*_siglip_b16.npz` (2.3G), `*_openclip_*_full.npz` (575M), `*_dinov2_*` (4.3G) | multi-GB | EXCLUDE_TOO_LARGE | regenerate via `scripts/extract_*` (downloads CLIP weights) |
| `realdata_validation/{features,maniskill/features}/*.npz` | ~1.7G | EXCLUDE_TOO_LARGE | OOD (App M) caches; regenerate |
| `realdata_validation/maniskill/videos/pair_*.npz` | ~292M | EXCLUDE_TOO_LARGE | raw OOD video arrays |

## Figures

| Path | Size | Matched item | Label | Reason |
|---|---|---|---|---|
| `figures_camera_ready/*.pdf,*.png,make_*.py` | 1.2M | Figs 2–6 + makers | INCLUDE_MAIN/APPENDIX | final figures + regenerators |
| `figures_new/`, `paper/figures/`, top-level `figure_*_v{1,2,3}.pdf`, `fig_cross_family*.pdf` | ~700K | older versions | EXCLUDE_DUPLICATE | superseded by camera-ready |
| `paper_updates/figures/*` | 108K | rebuttal experiments | EXCLUDE_OLD_DRAFT | not final-PDF figures |

## Paper / docs (NOT source of truth — used only to locate files)

| Path | Size | Label | Reason |
|---|---|---|---|
| `final_paper_snapshot/overleaf_final/EMNLP_final_overleaf.pdf` | 1.0M | INCLUDE_MAIN | the final PDF (anonymous); ships as `docs/EMNLP_final.pdf` |
| `paper/`, `paper_updates/` (LaTeX, old PDFs) | 1.5M | EXCLUDE_OLD_DRAFT | older drafts; not the selected final PDF |
| `LIB_DESIGN.md` | 16K | INCLUDE_OPTIONAL | architecture notes (App H) — scrub identity |

## Excluded private / unrelated / identity-risk

| Path | Label | Reason |
|---|---|---|
| `BLOCKING_ISSUES.md`, `NEGATIVE_RESULT_REPORT.md`, `NEGATIVE_RESULTS.md`, `DATA_GENERATION_PLAN.md`, `DATASET_*_PLAN.md`, `NEXT_ACTIONS.md`, `RESEARCH_PLAN_ACL2027.md`, `WEEKLY_REPORT_*.md`, `WEEKLY_TEMPLATE.md`, `FAREWELL_NOTE*.md`, `WAKE_UP_SUMMARY.md`, `STATUS.md`, `EXPERIMENT_*.md`, `IMPLEMENTATION_PLAN.md`, `PHASE_1_KICKOFF_PROMPT.md`, `REPO_AUDIT.md`, `E0_READY_FOR_REVIEW.md`, `ENV_DEPENDENCY_FIX.md`, `SIMULATOR_ENV_CHECK.md`, `RESULTS_SUMMARY.md`, `CLAIMS_MATRIX.md` | EXCLUDE_OLD_DRAFT / EXCLUDE_UNRELATED | planning notes, weekly reports, draft status — not the final PDF |
| `prompts/`, `review-stage/`, `research-wiki/`, `.aris/`, `verdicts/`, `findings/`, `phase0/` | EXCLUDE_PRIVATE | prompts, review strategy, private notes |
| `logs/`, `experiments/_orchestration/logs/`, `experiments/EXP-*/logs/`, `texput.log` | EXCLUDE_PRIVATE | logs (may contain server paths/usernames) |
| `.git/`, `.gitignore`, `__pycache__/`, `.claude/`, `.skills/`, `_tools/` | EXCLUDE_PRIVATE / EXCLUDE_UNRELATED | VCS / tooling / caches |
| any file containing author name / institution / personal email / absolute home or server paths / local usernames | EXCLUDE_IDENTITY_RISK (text -> scrub) | identity strings scrubbed on copy; binaries audited (see IDENTITY_LEAK_SCAN.md) |
