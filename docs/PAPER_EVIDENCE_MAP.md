# PAPER EVIDENCE MAP — CF-PrefBench v4 / LIB v0 anonymous artifact

Every figure/table/appendix detected in the final PDF gets one row. Paths are
**repo source paths** (under `/path/to/project/`); the artifact
copies a minimal subset (see `artifact_manifest.*`). Status legend:
`INCLUDE_MAIN`, `INCLUDE_APPENDIX`, `INCLUDE_OPTIONAL_CANDIDATE`, `EXCLUDE`,
plus `MISSING_SOURCE_DATA` (number reported in PDF, raw source file not located).

Common code dependencies (apply to nearly all rows, listed once):
- Model: `cf_pref_learning/models/lib.py` (+ alt-head variants `lib_hybrid/_residual/_multitask/_gumbel/_xl/_resid_pred.py`)
- Data/eval: `cf_pref_learning/data/*`, `cf_pref_learning/eval/*`
- Train: `scripts/train_lib.py`, `scripts/train_lib_v4_*.py`
- Feature extraction: `scripts/extract_clip_patch_features.py`, `extract_v4_new_axes_features*.py`, `extract_v4_new_axes_text_openclip_laion2b.py`
- Master numbers anchor: `outputs/phase5/MASTER_PAPER_DATA.tex`

---

## Main paper

| Item | Loc | Claim supported | Required result file | Required checkpoint | Required figure file | Status | Notes |
|---|---|---|---|---|---|---|---|
| Table 1 | §4.1 | B/32 motion-verb cliff (transit 0.500, scoot 0.679; cosine-correlated) | `outputs/phase5/exp3_aggregate/cliff_data_B32.json`; `experiments/EXP-G/seed{42,123,2024}/cliff_tokens_*.json` | EXP-B/B-B_published seeds 1,2,3 | — | INCLUDE_MAIN | |
| Table 2 | §4.2 | L/14 recovers cliff verbs to ≥0.905 | `outputs/phase5/exp3b_vitL14/L14_cliff_table.json` | exp3b_vitL14/checkpoints/lib_v0_vitL14_seed{1,2,3} | — | INCLUDE_MAIN | |
| Table 3 | §4.3 | 2×2 factorial: text-side swap (B-L 0.827) closes cliff | `outputs/phase5/exp4_axis_gen/cliff_table.json`; `experiments/EXP-B/{B-L,L-B,B-B_published,L-L_published}/seed*/cliff_tokens_*.json` | EXP-B/{B-L,L-B} seeds {1,42,123,2024} best.pt | `figures_camera_ready/fig_4_3_3axis_matrix_v2.*` (see fig README) | INCLUDE_MAIN | |
| Figure 2 | §4.3 | heatmap of 2×2 factorial | (same as Table 3) | (same) | `figures_camera_ready/fig_*` + `make_fig_4_3_3axis_matrix_v2.py` | INCLUDE_MAIN | regenerable from cliff_table.json |
| Table 4 | §4.4 | cross-family: cliff replicates, token identity swaps | `outputs/phase5/exp4_axis_gen/cliff_table.json`; `experiments/EXP-H/{OC-OC,SG-SG}/seed*/cliff_tokens_*.json` | EXP-H/{OC-OC,SG-SG} seeds {42,123,2024} | — | INCLUDE_MAIN | |
| Figure 3 | §4.4 | cross-family per-class bar chart | (same as Table 4) | (same) | `figures_camera_ready/fig_cross_family_v2.*` + `make_fig_cross_family_v2.py` | INCLUDE_MAIN | |
| Table 5 | §4.5 | 3-axis × 2-arch matrix, n=8, bootstrap CIs (SMALL 0.485, FAST 0.159) | `realdata_validation/expanded_tokens/tables/table_3axis_n16_paired.csv`; `outputs/phase5/exp4_axis_gen/cliff_table.json` | B/32 8-seed pool (EXP-F/B-32 + B-B_published) + L/14 8-seed pool (EXP-F/L-14 + exp3b_vitL14) | `figures_camera_ready/fig_4_4_mechanism_diagram_v2.*` | INCLUDE_MAIN | n=8 pool = `expanded_tokens/predictions/*` |
| Figure 4 | §4.5 | two mechanisms (motion r=0.627; size/speed bimodal) | `realdata_validation/expanded_tokens/tables/{motion_n12_for_fig_4a.csv,table_3axis_n16.csv}` | (Table 5 pool) | `figures_camera_ready/{fig_4_1_motion_cliff_b32_v2,fig_4_4_mechanism_diagram_v2}.*` + makers | INCLUDE_MAIN | |

## Appendix

| Item | Loc | Claim supported | Required result file | Required checkpoint | Required figure file | Status | Notes |
|---|---|---|---|---|---|---|---|
| Table 6 | App. A | substitution recovers transit 0.500→0.893, scoot 0.679→0.917 | `outputs/phase5/exp3_aggregate/cliff_data_B32.json` (substitution_recovery field); `experiments/EXP-G/seed*/cliff_tokens_*.json` | EXP-G seeds {42,123,2024}; probe code `scripts/train_lib_v4_with_sanity.py` | — | INCLUDE_APPENDIX | |
| Table 7 | App. C | per-token cliff, n=8, B/32 & L/14 | `realdata_validation/expanded_tokens/tables/{table_size_per_token.csv,table_speed_per_token.csv,per_seed_per_token.json}` | (Table 5 pools) | — | INCLUDE_APPENDIX | |
| Figure 5 | App. C | motion cliff B/32 vs L/14 (4 verbs) | `outputs/phase5/exp3b_vitL14/L14_cliff_table.json` | exp3b_vitL14 + B-B_published | `figures_camera_ready/fig_4_2_motion_dual_arch_v2.*` + maker | INCLUDE_APPENDIX | |
| Figure 6 | App. D | cliff on 3/3 viable heads (bilinear sig., MLP/X-attn marginal) | `experiments/EXP-D/{bilinear,mlp,crossattn}/seed{42,123,2024}/cliff_tokens_*.json` | EXP-D/{bilinear,mlp,crossattn} (+linear,clipscore omitted baselines) | `figures_camera_ready/fig_4_5_cross_method_v2.*` + maker | INCLUDE_APPENDIX | |
| Table 8 | App. D | numerical values for Fig 6 (gaps + p-values) | `experiments/EXP-D/*/seed*/cliff_tokens_*.json`; MASTER_PAPER_DATA.tex | (EXP-D ckpts) | — | INCLUDE_APPENDIX | |
| Table 9 | App. E | per-split sizes; grand total 7,749 | split jsonls in `data/cf_prefbench_v4/`; `data/raw/v4_new_axes/generation_summary.json`; MASTER_PAPER_DATA.tex | — | — | INCLUDE_APPENDIX | derivable by counting shipped splits |
| Table 10 | App. E | disjoint train/held-out paraphrase pools | `data/raw/v4_new_axes/generation_summary.json`; `scripts/generate_v4_new_axes.py`; MASTER_PAPER_DATA.tex | — | — | INCLUDE_APPENDIX | |
| Table 11 | App. F | 7-axis cliff incidence (cue-replacement vs wrapper) | `outputs/phase5/exp1_axes/REPORT.md` + jsons; MASTER_PAPER_DATA.tex | B-B_published (v4 axes) + v3 ckpts | — | INCLUDE_APPENDIX | |
| Table 12 | App. G | v3 action axis seen vs held-out (PFA→0) | `outputs/phase5/exp1_axes/` (action subdir) + REPORT.md | v3 action checkpoints (seeds 42/1337/2024) | — | INCLUDE_APPENDIX | uses `data/cf_prefbench/` (v3) |
| Table 13 | App. G | v3 per-held-out-instruction row accuracy | `outputs/phase5/exp1_axes/REPORT.md` | (v3 action ckpts) | — | INCLUDE_APPENDIX | |
| Table 14 | App. I.2 | probe applicability per method class | MASTER_PAPER_DATA.tex; `scripts/train_lib_v4_with_sanity.py`, `scripts/sanity_eval_hybrid.py` | — | — | INCLUDE_APPENDIX | matrix is structural, code-derived |
| Table 15 | App. I.3 | Phase-3 hybrid sanity (SC-4 collapses, SC-5 doesn't) | `outputs/phase5/exp6_cross_sanity/*`; sanity jsons; MASTER_PAPER_DATA.tex | hybrid ckpt (OPTIONAL) | — | INCLUDE_APPENDIX | hybrid checkpoint optional |
| Table 16 | App. I.3 | pure-LIB SC-6 zero/shuffle + substitution (B/32 0.750→0.893) | `outputs/phase5/exp2_motion_verify/sc6/sanity_v4_seed1.json` | exp2_motion_verify/sc6/lib_v0_seed1 | — | INCLUDE_APPENDIX | |
| Table 17 | App. I.3 | Phase-4 anti-collapse (no variant passes targets) | `outputs/phase5/exp2_motion_verify/*`; MASTER_PAPER_DATA.tex | anti-collapse ckpts | — | INCLUDE_APPENDIX | negative result, REPORTED in PDF → included |
| Table 18 | App. K | cross-family per-token (B-B/OC-OC/SG-SG) | `experiments/EXP-H/{OC-OC,SG-SG}/seed*/cliff_tokens_*.json`; `outputs/phase5/exp4_axis_gen/cliff_table.json` | EXP-H ckpts | — | INCLUDE_APPENDIX | |
| Table 19 | App. L | zero-shot GPT-4o / Qwen2.5-VL per-class | `outputs/phase5/exp5_vlm/{gpt4o,qwen}/*.json` + predictions jsonl | none (API/zero-shot) | — | INCLUDE_APPENDIX | scripts: run_gpt4o_judge_v4_cliff.py, run_qwen2vl_judge_v4_cliff.py, build_vlm_table.py |
| Appendix B | App. B | n=12 verb pool + confound controls | `realdata_validation/expanded_tokens/tables/motion_n12_for_fig_4a.csv`; `make_expanded_cliff_v3.py` | (B/32 pool) | — | INCLUDE_APPENDIX | |
| Appendix H | App. H | LIB v0 architecture details | `cf_pref_learning/models/lib.py`; `LIB_DESIGN.md` (anonymize) | — | Figure 1 (PDF p.3) | INCLUDE_APPENDIX | |
| Appendix J | App. J | label balance + uninformative-text probe | `outputs/phase5/exp2_motion_verify/*` sanity jsons | — | — | INCLUDE_APPENDIX | single-seed caveat noted in PDF |
| Figure 1 | §3/App.H | LIB v0 architecture diagram | — | — | `docs/EMNLP_final.pdf` p.3 (hand-drawn) | INCLUDE_APPENDIX | no standalone vector source; diagram only |
| Appendix M | §4.7 | CLEVRER chance (~0.49); ManiSkill FAST 0.33/SLOW 0.62, substitution→0.64; L/14 null | numbers in PDF + MASTER_PAPER_DATA.tex; feature caches `realdata_validation/{features,maniskill}` (~1.7 GB, too large); qualitative videos `outputs/robot_qualitative_maniskill*` | published LIB ckpts (zero-shot) | — | **MISSING_SOURCE_DATA / WARNING** | OOD per-trial prediction JSONs + eval scripts not located as discrete files; feature caches EXCLUDE_TOO_LARGE. Documented in REPRODUCTION_GUIDE + DATASET_CARD; numbers recorded in docs. Not a main-paper item → WARNING, not NOT-READY. |

---

## Coverage summary

- **Main items (Tables 1–5, Figures 1–4):** all have INCLUDE-status code + result + (where numeric) figure source → no main-paper WARNING.
- **Appendix items (Tables 6–19, Figures 5–6, Apps A–L):** all INCLUDE_APPENDIX with located evidence.
- **Single WARNING:** Appendix M (OOD) — per-trial prediction files / eval scripts not recoverable; reported numbers preserved in PDF + `MASTER_PAPER_DATA.tex`; raw feature caches excluded for size. Disclosed honestly in the final report and reproduction guide.
- **Reported negative results are INCLUDED** (Table 15 SC-4 collapse, Table 17 anti-collapse failures, Appendix M CLEVRER visual-domain gap) because the final PDF reports them.
- **Excluded** (not in final PDF): old draft reports, weekly reports, blocking-issues notes, prompts, review-stage, research-wiki, farewell notes — see `REPO_FILE_CLASSIFICATION.md`.
