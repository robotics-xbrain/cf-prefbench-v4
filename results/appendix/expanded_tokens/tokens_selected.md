# Stage B — Selected New Held-Out Tokens

Goal: expand each (axis, class) cell from 2 → 4 tokens. We add **8 new tokens** (2 per class)
to the existing 8 size+speed cliff tokens, yielding 16 tokens × 8 seeds = **n=128** datapoints
per axis (size and speed each).

## Methodology

For every candidate, we compute `cos_to_train` exactly as the existing eval pipeline does
(`experiments/EXP-B/eval_crossenc_on_cliff_tokens.py`):

- Token mean = mean of the candidate's text features across its unique held-out instructions
  (instructions are produced by string-substituting the candidate token into the same source
  rows used by `scripts/generate_v4_axis_gen_tests.py`).
- Train mean = axis-level mean of training-instruction text features from
  `data/cf_prefbench_v4/train.jsonl`.
- Cosine is on the **unnormalized** CLIP text features, matching the existing pipeline.

Cliff zone target: `[0.87, 0.97]`.

## Training-token guardrails

I scanned `train.jsonl` to confirm which words actually appear at training time:

- **SIZE training tokens**: `small` (264 rows), `large` (144), `big` (120)
- **SPEED training tokens**: `quickly` (264), `slowly` (264)

All 8 selected new tokens are absent from `train.jsonl`. (The generator constants
`SIZE_SMALL = {tiny, small}`, `SPEED_FAST = {quickly, rapidly}`, etc. in
`scripts/generate_v4_axis_gen_tests.py` are *substitution-target* sets, not training-token
sets — `tiny`, `rapidly`, `leisurely` listed there are NOT in training data.)

## Selected 8 new tokens

| Axis  | Class | Token        | B/32 cos | L/14 cos | Cliff zone (B/32) | Cliff zone (L/14) | # unique instr |
|-------|-------|--------------|----------|----------|-------------------|-------------------|----------------|
| size  | BIG   | enormous     | 0.9585   | 0.9467   | ✓ in              | ✓ in              | 15 |
| size  | BIG   | vast         | 0.9603   | 0.9310   | ✓ in              | ✓ in              | 15 |
| size  | SMALL | tiny         | 0.9594   | 0.9483   | ✓ in              | ✓ in              | 18 |
| size  | SMALL | minute       | 0.9406   | 0.9239   | ✓ in              | ✓ in              | 18 |
| speed | FAST  | swiftly      | 0.9652   | 0.9498   | ✓ in              | ✓ in              | 3  |
| speed | FAST  | hastily      | 0.9662   | 0.9324   | ✓ in              | ✓ in              | 3  |
| speed | SLOW  | leisurely    | 0.9630   | 0.9334   | ✓ in              | ✓ in              | 3  |
| speed | SLOW  | languidly    | 0.9598   | 0.9237   | ✓ in              | ✓ in              | 3  |

### Existing 8 tokens (paper) — for reference

| Axis  | Class | Token        | B/32 cos | L/14 cos |
|-------|-------|--------------|----------|----------|
| size  | BIG   | colossal     | 0.9559   | 0.9292   |
| size  | BIG   | gigantic     | 0.9588   | 0.9460   |
| size  | SMALL | miniature    | 0.9489   | 0.9347   |
| size  | SMALL | petite       | 0.9446   | 0.9357   |
| speed | FAST  | briskly      | 0.9568   | 0.9369   |
| speed | FAST  | speedily     | 0.9609   | 0.9231   |
| speed | SLOW  | sluggishly   | 0.9539   | 0.9076   |
| speed | SLOW  | gradually    | 0.9578   | 0.9341   |

## Notes

- **Original primary `rapidly` was swapped for `hastily`** (decision D-004). `rapidly` sat at
  B/32 = 0.9731 (+0.003 above the cliff-zone upper bound 0.97). `hastily` lands inside the
  zone on both backbones (B/32 = 0.9662, L/14 = 0.9324). Same FAST semantic class, no
  training-token overlap.
- All 8 new tokens have **same-or-higher cosines on B/32 than the existing cliff tokens** —
  if the cliff is robust, we should still see FAST near zero (matching the entrenched
  SLOW-default class bias rather than a cosine threshold).
- New speed tokens have only 3 unique instructions each (matching the existing speed test
  set's instruction diversity: speed source uses {shift, advance, translate} × adverb).
  The accuracy denominator is **42 rows per token** (3 unique instructions × ~14 video
  pairs), identical to existing tokens.

## Sample instructions

| Token        | Sample instruction                       |
|--------------|------------------------------------------|
| enormous     | `fetch the enormous purple block`        |
| vast         | `fetch the vast purple block`            |
| tiny         | `fetch the tiny yellow block`            |
| minute       | `fetch the minute yellow block`          |
| swiftly      | `shift the block swiftly`                |
| hastily      | `shift the block hastily`                |
| leisurely    | `shift the block leisurely`              |
| languidly    | `shift the block languidly`              |

Source: `realdata_validation/expanded_tokens/scripts/cosines_{b32,l14}.json`
