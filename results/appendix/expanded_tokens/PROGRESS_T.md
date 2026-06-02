# PROGRESS_T — Expanded Token Pool (n=4 per class)

Branch: `emnlp_expandedtokens_20260524` (off `rebuttal-experiments`)
Started: 2026-05-24
GPU: CUDA_VISIBLE_DEVICES=1

## Stage A — Setup + sanity check
- [x] Located checkpoints: 3 published (seed 1/2/3) + 5 EXP-F (seed 7/31/99/256/2025) = **n=8 seeds per backbone**.
  - B/32: `outputs/phase5/exp{2_motion_verify,3_4th_verb}/checkpoints/lib_v0_seed{1,2,3}_best.pt` + `experiments/EXP-F/B-32/seed{7,31,99,256,2025}/lib_v0_b32_efg_seed*_best.pt`
  - L/14: `outputs/phase5/exp3b_vitL14/checkpoints/lib_v0_vitL14_seed{1,2,3}_best.pt` + `experiments/EXP-F/L-14/seed{7,31,99,256,2025}/lib_v0_l14_efg_seed*_best.pt`
- [x] Read `experiments/EXP-B/eval_crossenc_on_cliff_tokens.py` — uses per-token JSONLs, computes per-token accuracy + cos_to_train.
- [x] Read `scripts/generate_v4_axis_gen_tests.py` — confirms the generation template: substitute the new token into existing held-out JSONL rows in the same semantic class.
- [x] Reproduced n=8 published-cliff numbers in-place:
  - briskly: n=8 mean = 0.0298 (paper n=3 reported 0.048 — consistent under wider seed pool)
  - FAST class n=8 ≈ 0.092 (paper n=3 reported 0.115)
- [x] Verified actual training tokens (vs. generator constants):
  - SIZE: only `small`, `large`, `big` appear in train.jsonl
  - SPEED: only `quickly`, `slowly` appear in train.jsonl
  - User-proposed `tiny`, `rapidly`, `leisurely` are NOT in training → safe to use as held-out

## Stage B — Token selection + cosine validation
- [x] Computed B/32 + L/14 cosines for 8 primary + 9 backup candidate tokens.
- [x] All 8 user-proposed primary tokens stay (one borderline: `rapidly` B/32 = 0.9731, +0.003 above 0.97 — user explicitly authorized inclusion).
- [x] Wrote `tokens_selected.md` (markdown table).
- [x] **STOPPED for user review** before running Stage C inference. To resume: green-light the table and proceed.

## Stage C — Inference
- [x] Built 8 new heldout JSONLs (42 rows each, matching paper protocol). `data/cf_prefbench_v4/test_heldout_{size,speed}_{enormous,vast,tiny,minute,swiftly,hastily,leisurely,languidly}.jsonl`.
- [x] Ran new-token eval over 8 B/32 + 8 L/14 seeds = 16 checkpoints; outputs in `predictions/{b32,l14}_seed{S}_new_tokens.{json,csv}` (16 JSONs, 16 CSVs).
- [x] Total inference time: ~8 minutes on GPU 1.

## Stage D — Stats + tables
- [x] Per-token n=8-seed mean + 95% bootstrap CIs (1000 resamples).
- [x] Class-level table_3axis_n16.csv with unpaired (token×seed) CIs.
- [x] Class-level table_3axis_n16_paired.csv with paired-by-seed CIs.
- [x] Surprise tokens flagged at thresholds (cliff_fail ≤ 0.50, ceiling_pass ≥ 0.75).
- [x] Per-class counts and 5 surprise tokens identified (miniature/petite/tiny SMALL borderline; gradually/leisurely SLOW borderline).

## Stage E — LaTeX patches
- [x] `patch_T01_table_3axis_cliff_n8.tex` — replacement for tab:3axis-cliff (n=4 tokens × 8 seeds, paired-by-seed CIs)
- [x] `patch_T02_table_per_token.tex` — replacement for tab:per-token-cliff (20-token expanded table)
- [x] `patch_T03_section_45_text.tex` — replacement for §4.3 preamble ("eight new" → "sixteen new" + token list)
- [x] `patch_T04_section_45_narrative.tex` — added 2-3 sentence narrative on within-class heterogeneity and surprise tokens
- [x] `STAGE_T_REPORT.md` — final diagnosis with top-line, per-class verdict, implications, reviewer-facing calibration
