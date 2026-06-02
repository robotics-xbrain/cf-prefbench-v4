# Figures — file → final-PDF figure mapping

The authoritative figures are in **`docs/EMNLP_final.pdf`** (the submitted PDF, the
sole source of truth). Camera-ready source filenames here use the *older* section
numbering (`fig_4_x`); the final PDF reorganized figure order, so the mapping below
is by **content** and is the best available reconstruction. Generators are in
`scripts/figures/` (they read the original repo's numbers); `scripts/reproduce_figures.sh`
re-renders Figures 2–4 from the released `results/`.

| Final PDF figure | File in this artifact | Source / note |
|---|---|---|
| Figure 1 — LIB v0 architecture | *(none)* | hand-drawn TikZ; see `docs/EMNLP_final.pdf` p.3 |
| Figure 2 — 2×2 encoder factorial heatmap | `figures/main/figure2_2x2_encoder_factorial.pdf` | from `paper_updates` `exp_b_2x2_factorial` |
| Figure 3 — cross-family validation | `figures/main/figure3_cross_family.{pdf,png}` | `make_fig_cross_family_v2.py` |
| Figure 4 — two cliff mechanisms (a)+(b) | `figures/main/figure4_two_mechanisms.{pdf,png}` | `make_fig_4_4_mechanism_v2.py` |
| Figure 5 — motion cliff under two backbones | `figures/appendix/figure5_motion_two_backbones.{pdf,png}` | `make_fig_4_2_dual_arch_v2.py` |
| Figure 6 — alternative-head robustness | `figures/appendix/figure6_alternative_heads.{pdf,png}` | `make_fig_4_5_cross_method_v2.py` |

Supplementary (earlier panels / appendix variants, content-related but not 1:1 to a
final figure number):
- `figures/appendix/supp_motion_cliff_b32_scatter.pdf` — motion cosine scatter (Fig 4a panel)
- `figures/appendix/supp_3axis_matrix.pdf` — three-axis × two-arch matrix (Table 5 viz)
- `figures/appendix/supp_expanded_cliff_B-B.pdf` — expanded-pool cliff (Appendix B/C)

`figures/reproduced/` — figures re-rendered from released numbers by
`scripts/reproduce_figures.sh` (independent re-derivation; styling differs from
camera-ready).
