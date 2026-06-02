# DECISIONS_T — Expanded Token Pool

## D-001 (Stage A): Use existing n=8 checkpoint pool
The published paper claims `n=6 per cell` in Table tab:3axis-cliff (3 seeds × 2 tokens). But
EXP-F has already extended this to **n=8 seeds**, used in the camera-ready figure pipeline.
This task therefore expands along the *token* dimension (2 → 4 per class) while keeping the
same 8-seed pool, yielding n=8 seeds × 4 tokens = **n=32 per class cell** for n=16 new tokens
total (8 new size + 8 new speed). Wait — the user asked for **2 → 4 per class** which is +2
per class × 4 classes (size BIG/SMALL, speed FAST/SLOW) = **8 new tokens** total, not 16.
Re-confirm: each class goes from 2 → 4 tokens; total new tokens = 4 classes × 2 new = **8**.
Combined with the 8 existing tokens = 16 total cliff tokens per axis-class evaluation.

## D-002 (Stage A): Training-token check vs. generator constants
The generator script's `SIZE_SMALL = {tiny, small}` listed `tiny` as if it were a training
token, but a direct scan of `train.jsonl` shows `tiny` does **not** appear in training data —
only `small` is the SMALL training adjective. Similarly `rapidly` (FAST) and `leisurely`
(SLOW) listed as "training" in the generator are also NOT in `train.jsonl`. The generator's
sets are *semantic-class membership* lists used for substitution, not training-token lists.
**Actual training tokens**:
- SIZE: `small`, `large`, `big` (BIG = large/big; SMALL = small)
- SPEED: `quickly`, `slowly` (FAST = quickly; SLOW = slowly)

All 8 user-proposed new tokens (enormous, vast, tiny, minute, rapidly, swiftly, leisurely,
languidly) are confirmed held-out from training.

## D-004 (post-Stage B, user-confirmed): Swap `rapidly` → `hastily`
After Stage B review, user requested swapping FAST candidate `rapidly` (B/32 = 0.9731,
+0.003 above the [0.87, 0.97] cliff zone) for `hastily` (B/32 = 0.9662, L/14 = 0.9324) —
solidly inside the zone, same FAST semantic class, no train-set overlap. Final FAST class
pool: briskly, speedily, swiftly, hastily.

## D-003 (Stage B): Cosine validation methodology
Following `eval_crossenc_on_cliff_tokens.py`: per-token "cos_to_train" is the cosine between
- the token's *mean* text feature (across all instructions in its held-out JSONL), and
- the *mean* training-instruction feature (axis-level: mean over all train.jsonl rows whose
  axis matches the token's axis).

Cliff zone target: cosine in `[0.87, 0.97]`. Tokens outside this zone are recorded but still
included in the expanded pool — the user explicitly opted to "记录但仍用作 expanded pool 的
一部分" if a candidate cosine falls outside.
