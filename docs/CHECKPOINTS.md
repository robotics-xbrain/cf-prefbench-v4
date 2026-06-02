# Checkpoints — map to paper tables/figures

All are LIB v0 (or alternative-head) `*_best.pt` learned heads over **frozen** CLIP
features. Third-party encoder weights are NOT included. 48 checkpoints, ~131 MB.
Intermediate `ckpt_epoch_*.pt` were excluded (best.pt suffices).

| Dir / file group | Backbone (visual / text) | Seeds | Backs |
|---|---|---|---|
| `checkpoints/lib_b32/lib_v0_b32_seed{1,2,3,7,31,99,256,2025}_best.pt` | CLIP B/32 / B/32 (= **B-B** cell) | 8-seed pool {1,2,3,7,31,99,256,2025} | Tables 1,2,3,5,6,7,11; Fig 4,5 |
| `checkpoints/lib_l14/lib_v0_l14_seed{1,2,3,7,31,99,256,2025}_best.pt` | CLIP L/14 / L/14 (= **L-L** cell) | 8-seed pool | Tables 2,3,5,7; Fig 4,5 |
| `checkpoints/factorial/lib_v0_B-L_seed{1,42,123,2024}_best.pt` | B/32 visual / L/14 text | {1,42,123,2024} (paper uses 42/123/2024) | Table 3, Fig 2 |
| `checkpoints/factorial/lib_v0_L-B_seed{42,123,2024}_best.pt` | L/14 visual / B/32 text | {42,123,2024} | Table 3, Fig 2 |
| `checkpoints/cross_family/lib_v0_OC-OC_seed{42,123,2024}_best.pt` | OpenCLIP-LAION-2B (both) | {42,123,2024} | Table 4,18; Fig 3 |
| `checkpoints/cross_family/lib_v0_SG-SG_seed{42,123,2024}_best.pt` | SigLIP-WebLI (both, 768→512 pre-proj) | {42,123,2024} | Table 4,18; Fig 3 |
| `checkpoints/alt_heads/{bilinear,mlp,crossattn}_seed{42,123,2024}_best.pt` | B/32 (alt heads) | {42,123,2024} | Fig 6, Table 8 |
| `checkpoints/alt_heads/{linear,clipscore}_seed{42,123,2024}_best.pt` | B/32 (omitted baselines) | {42,123,2024} | Fig 6 (degenerate baselines) |
| `checkpoints/substitution/lib_v0_subst_seed{42,123,2024}_best.pt` | B/32 | {42,123,2024} | Table 1,6 (substitution probe) |
| `checkpoints/sanity/lib_v0_motion_sanity_seed1_best.pt` | B/32 | 1 | Table 16 (pure-LIB sanity) |

Notes:
- The **B-B** (baseline) and **L-L** (full upgrade) cells of the 2×2 factorial (Table 3)
  reuse the `lib_b32` and `lib_l14` main checkpoints — there is no separate B-B/L-L
  checkpoint directory.
- The canonical B/32 seeds {1,2,3} are `lib_b32/lib_v0_b32_seed{1,2,3}_best.pt`
  (sourced from the published phase-5 runs); seeds {7,31,99,256,2025} extend to the
  n=8 pool used by Tables 5/7.
- To evaluate: regenerate the matching feature cache (`scripts/eval/extract_*`) then run
  `scripts/eval/evaluate_lib_on_new_tokens.py` / `evaluate_lib_on_scoot.py`.
- Per-seed evaluation outputs (the numbers these checkpoints produced) are in
  `results/metrics/*.json` (`class_aggregates`) so tables reproduce without re-evaluation.
