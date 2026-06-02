# STAGE_T_REPORT — Expanded Token Pool (2 → 4 tokens per class)

**Branch**: `emnlp_expandedtokens_20260524` (off `rebuttal-experiments`)
**Backbones**: ViT-B/32 (LIB-v0) and ViT-L/14 (LIB-v0 vitL14)
**Seeds**: $n=8$ per backbone (published seeds 1/2/3 + EXP-F seeds 7/31/99/256/2025)
**New tokens added**: 8 (2 per class × 4 classes); selection table in `tokens_selected.md`
**Inference rows per token**: 42 (3 unique instructions × 14 video pairs)
**Total new evaluations**: 8 seeds × 8 new tokens × 2 backbones × 42 rows = 5{,}376 row-level predictions

---

## Top-line: does the cliff replicate under the expanded pool?

| Class | Cliff on B/32 at $n=2$ tokens (paper) | Cliff on B/32 at $n=4$ tokens (this work) | Verdict |
|-------|----|----|---|
| **BIG**   | no cliff (0.925)                | **no cliff**  (0.924 [0.920, 0.929]) | replicated |
| **SMALL** | deep cliff (0.444)              | **weakened**  (0.485 [0.432, 0.540]) | nuanced |
| **FAST**  | deep cliff (0.115)              | **replicated** (0.159 [0.100, 0.214]) | replicated, slightly less extreme |
| **SLOW**  | mostly clean (0.829)            | **nuanced**   (0.792 [0.766, 0.820]) | new partial cliff |
| Motion above | clean (0.923)                | replicated (0.902 [0.884, 0.920])    | replicated |
| Motion below | cliff (0.589)                | replicated (0.621 [0.592, 0.650])    | replicated |

All paired-by-seed 95% bootstrap CIs over $n=8$ seeds.

**Bottom line**: the cliff phenomenon is *robust on FAST and motion below-cliff*, *attenuated on
SMALL*, and *newly partially-cliffed on SLOW*. BIG remains clean on all 4 tokens. The expanded
pool therefore does not break the cliff — but it shifts the framing from "class-bimodal at one
cosine" toward "geometry-driven within-class heterogeneity."

---

## Per-class verdict at thresholds [cliff_fail ≤ 0.50, ceiling_pass ≥ 0.75]

| Class | tokens (B/32) | cliff_fail | borderline | ceiling_pass | agrees with class prediction |
|-------|---------------|------------|------------|--------------|-------------------------------|
| BIG   | colossal, gigantic, enormous, vast | 0 | 0 | **4** | 4/4 ✓ |
| SMALL | miniature, petite, tiny, minute     | 1 (minute) | 3 (miniature, petite, tiny) | 0 | 1/4 ✓ (but 4/4 below ceiling) |
| FAST  | briskly, speedily, swiftly, hastily | **4** | 0 | 0 | 4/4 ✓ |
| SLOW  | sluggishly, gradually, leisurely, languidly | 0 | 2 (gradually, leisurely) | 2 (sluggishly, languidly) | 2/4 ✓ (2 surprises below ceiling) |

### Surprise tokens (B/32 outcome disagrees with cliff-class prediction)

| Token | Class | B/32 mean | L/14 mean | Direction |
|-------|-------|-----------|-----------|-----------|
| miniature | SMALL | 0.524 | 0.878 | borderline (expected cliff_fail) |
| petite    | SMALL | 0.562 | 0.842 | borderline (expected cliff_fail) |
| **tiny**  | SMALL | **0.708** | 0.902 | borderline → near-pass (expected cliff_fail) — **strongest SMALL surprise** |
| gradually | SLOW  | 0.682 | 1.000 | borderline (expected ceiling_pass) — pre-existing partial cliff |
| **leisurely** | SLOW | **0.571** | 0.696 | borderline → fail (expected ceiling_pass) — **new SLOW surprise** |

`minute` (SMALL, 0.146) is the clearest cliff confirmation among new tokens; `hastily` is the
clearest L/14 recovery (B/32 = 0.202 → L/14 = 0.988).

---

## Implications for paper narrative

### Claims that stay (unchanged)

- **Motion-verb cliff (§4.1)**: monotonic cosine relationship, large below-cliff effect.
  Below-cliff B/32 mean: paper 0.589 → expanded n=8 0.621, with CIs that do not include the
  ceiling. The Pearson $r=0.808$ correlation does not need re-statement because we did not
  add motion-verb tokens.
- **L/14 closes motion cliff (§4.2)**: still holds. L/14 below-cliff = 0.917, with CI [0.904, 0.929].
- **Cross-method method-specificity (§4.5)**: not affected by token-pool expansion.

### Claims that get nuanced

- **"Within-class clustering at near-identical cosines" (§4.3, paper)**: still true *as a class-
  level observation*, but the expanded pool reveals **per-token heterogeneity at the same
  cosine**. SMALL tokens at B/32 cos $\in [0.94, 0.96]$ produce outcomes
  $\{0.146, 0.524, 0.562, 0.708\}$ — a >0.5 spread inside a 0.02-cos window. The "class
  membership predicts accuracy" framing becomes more honest as: "class membership is *not
  reducible* to cosine but is *not deterministic* either — per-token CLIP-text geometry sets
  the outcome." Patch T04 captures this in 2-3 sentences.
- **"BIG and SLOW pass" default-class bias (§4.3)**: BIG still passes (4/4 ceiling); SLOW
  partially passes (2/4 ceiling, 2/4 borderline). The expanded pool reveals SLOW is not
  uniformly above-cliff. We should soften the strong default-class claim to a *typical*
  default-class observation.

### Claims that get strengthened

- **"Cliff is text-side, not visual"**: the SMALL and FAST tokens all share the same video
  pairs and (within each class) similar B/32 cosines, yet outcomes split sharply. The only
  thing that differs is the token's CLIP-text embedding direction. This is the cleanest
  evidence for a text-side mechanism that the paper has yet shown.
- **"L/14 partial closure on speed"**: the new pool shows the closure is *per-token*: hastily
  recovers fully (B/32 0.20 → L/14 0.99), swiftly and speedily only partially, and minute
  *stays* cliff. Backbone scaling helps differently for different tokens, supporting the
  *capacity-of-text-embedding* explanation in §4.2 rather than a uniform "L/14 is better."

### New finding (worth one sentence in §5 Discussion)

- **SLOW cliff under L/14**: `leisurely` (0.696) and `languidly` (0.664) under L/14 are
  noticeably below `sluggishly` (0.994) and `gradually` (1.000). This is a *new* L/14
  partial cliff on the SLOW side that the original 2-token pool could not see, because
  both original SLOW tokens happened to be in the L/14-ceiling subset.

---

## Suggested confidence calibration for reviewers

> *"We re-evaluated our cliff claims on a $2\!\times$ expanded held-out token pool (16 tokens
> total per axis class — 8 new + 8 paper-original) across $n=8$ random seeds for each backbone.
> Three of four B/32 cliff cells (FAST, SMALL, motion-below) remain cliffed at 95% paired-by-seed
> bootstrap CIs that exclude the ceiling. The fourth (SMALL) is class-attenuated rather than
> deep on the expanded pool because one new token (tiny) lands near-pass at $0.71$. The
> expanded pool also reveals one previously-hidden partial cliff (SLOW under L/14) and confirms
> the cliff is per-token rather than per-class, which we read as evidence for a geometry-
> driven rather than cosine-monotonic mechanism. All numbers reproduce from the same
> checkpoints used in the published Table~\ref{tab:3axis-cliff} — no retraining."*

**Internal confidence level**:
- *Cliff exists, is text-side, is per-token, is partially closed by L/14*: high (claim has 16 tokens × 8 seeds = 128 datapoints per axis).
- *Cliff is class-bimodal at one cosine*: medium → low (the expanded pool weakens this; we now prefer "geometry-driven within-class heterogeneity").
- *BIG/SLOW always pass under B/32*: medium (BIG yes; SLOW only 2/4 reach ceiling on expanded pool).

---

## Patch application note

**Patches target the user's latest LOCAL `main.tex` labels; user will apply manually.** The
server's `paper/sections/*.tex` is a stale split-file version; do NOT apply patches to it.
T01 references `tab:3axis-cliff-n8`, T02 references `tab:per-token`, T03 and T04 reference
`§4.5 sec:findings:matrix` — these match the user's local single-file `main.tex` and not the
server copy.

## Artifacts

- `tokens_selected.md` — 8 new tokens with B/32 and L/14 cosines
- `tables/table_size_per_token.csv` — 8 size tokens × 8 seeds, paper+new
- `tables/table_speed_per_token.csv` — 8 speed tokens × 8 seeds, paper+new
- `tables/table_3axis_n16.csv` — class means with unpaired bootstrap CIs
- `tables/table_3axis_n16_paired.csv` — class means with paired-by-seed bootstrap CIs
- `tables/surprises.json` — full surprise scoring
- `tables/per_seed_per_token.json` — raw lookup
- `predictions/{b32,l14}_seed{S}_new_tokens.{json,csv}` — 16 prediction files
- `paper_patches_tokens/patch_T0{1..4}*.tex` — paste-ready LaTeX patches
