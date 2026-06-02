# Dataset card — CF-PrefBench v4

## Summary
A controlled compositional **preference** benchmark. Each row is a triple
`(video_A, video_B, instruction)` differing on exactly one binding axis; the label
`y ∈ {A, B}` is the video matching the instruction. Produced by a **deterministic
rendered simulator** (24-frame trajectories, 192×144). Minimal rendering removes
photorealism/clutter so any failure localises to the binding mechanism, not visual
recognition.

## Axes (7 binding + 1 diagnostic)
`color, object, action, spatial, size, motion_sequence, speed` + `impossible_premise`.

## Splits and sizes (Table 9; verified by counting shipped files)
Per binding axis: train **528**, val **84**, test_seen **72**,
test_heldout_lexical **84**, test_heldout_camera **72**, test_heldout_color **84**,
test_heldout_spatial **84**, test_hard_negatives **72** → **1,080/axis**.
`impossible_premise`: **27** rows in each non-train split, 0 train.
**Grand total = 7×1,080 + 7×27 = 7,749 core rows.**

The **critical** split is `test_heldout_lexical`: instructions use cue variants that
never appear in training, chosen so held-out cue variants span CLIP-text cosine
`[0.87, 0.97]` to training (on ViT-B/32). On `motion_sequence` a 4th held-out verb
`scoot` probes the transition zone.

## Files in this artifact
- `data/cf_prefbench_v4/*.jsonl` — the 8 core splits.
- `data/cf_prefbench_v4/test_heldout_{size,speed}_*.jsonl` — 16 expanded per-token
  test splits (colossal/gigantic/enormous/vast, miniature/petite/tiny/minute,
  briskly/speedily/swiftly/hastily, sluggishly/gradually/leisurely/languidly).
- `data/cf_prefbench_v4/test_heldout_lexical_scoot.jsonl` — scoot probe.
- `data/cf_prefbench_v3/*.jsonl` — v3 (4-axis) benchmark for the Appendix G action
  replication (Tables 12–13).
- `data/raw_provenance/*.json` — generation summaries (axes, paraphrase pools, split
  distributions) backing Tables 9–10.
- `data/samples/` — 6 sample `.mp4` trajectories (full video set excluded for size).

## Paraphrase pools (Table 10, disjoint by construction)
e.g. motion_sequence train `{drag, move, push}` → held-out `{convey, shift, transit, scoot}`;
size train `{grasp, lift, pick}` → held-out `{fetch, retrieve, secure}`; speed train
`{carry, move, translate}` → held-out `{advance, shift, transport}`; action (v3) uses
5 train / 4 held-out verbs. Full pools in `data/raw_provenance/v4_new_axes_generation_summary.json`
and `scripts/generate_v4_new_axes.py`.

## Row schema
JSONL; fields include `axis`, the video pair references, instruction text, label,
`pair_id` / `flip_group` for counterfactual grouping. See
`cf_pref_learning/data/schemas.py`.

## Regenerating the full dataset (videos excluded for size)
`python scripts/generate_v4_new_axes.py` (v4 axes), `scripts/generate_v3_anti_shortcut_cf_prefbench.py`
(v3), `scripts/generate_v4_axis_gen_tests.py` (expanded per-token tests). Expected size:
raw videos ≈ 62 MB (1,080 v4 + 1,566 v3 clips), metadata/splits ≈ 10 MB.

## Provenance, ethics, limitations
Labels are derived deterministically from simulator state — **no human annotation**,
no personal data, no demographic aggregation (see paper Ethics). Limitation: whether
human annotators would produce the same labels, and whether the cliff transfers to
human-labeled data, is open. Out-of-domain rendered validation (CLEVRER, ManiSkill)
is reported in Appendix M; its per-trial artifacts are not shipped (see reproduction
guide).
