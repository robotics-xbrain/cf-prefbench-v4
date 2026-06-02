# data/

- `cf_prefbench_v4/` — CF-PrefBench v4: 8 core split JSONLs (7 axes + impossible_premise,
  7,749 rows) + 16 expanded per-token size/speed test splits + scoot probe.
- `cf_prefbench_v3/` — v3 (4-axis) benchmark for the Appendix G action-axis replication.
- `raw_provenance/` — generation summaries (axes, paraphrase pools, split distributions)
  backing Tables 9–10.
- `samples/` — 6 sample `.mp4` trajectories. The full raw video set (~62 MB) is excluded;
  regenerate with `scripts/generate_v4_new_axes.py`. See `docs/DATASET_CARD.md`.
