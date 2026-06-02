# results/

- `MASTER_PAPER_DATA.tex` — committed single source of truth for every paper number.
- `tables/` — aggregated result JSONs behind Tables 1–7, 18.
- `metrics/` — per-seed raw cliff metrics (`class_aggregates`) from EXP-B/D/F/G/H;
  tables reproduce from these without re-evaluating checkpoints.
- `appendix/` — VLM (Table 19), 7-axis + v3 action (Tables 11–13), sanity battery
  (Tables 15–17, App J), expanded-token n=8 pool (Tables 5/7, App B/C), section notes.
- `reproduced_tables/` — output of `scripts/reproduce_main_tables.sh`.

Regenerate: `bash scripts/reproduce_main_tables.sh`.
