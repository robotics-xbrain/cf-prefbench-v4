# cf_pref_learning/

Importable Python package for CF-PrefBench / LIB v0.
- `models/` — LIB v0 (`lib.py`) + alternative heads (`lib_hybrid/_residual/_multitask/
  _gumbel/_xl/_resid_pred.py`) + baseline preference head.
- `data/` — CF-PrefBench builder, schema, splitting, converters, leakage audit.
- `eval/` — metrics (per-row accuracy, PFA), bootstrap CIs, impossible-premise metric.
- `train/`, `utils/`, `downstream/` — training harness stubs and IO helpers.

`python -c "from cf_pref_learning.models.lib import LIBModule; LIBModule()"` should work
once torch is installed. See `docs/MODEL_CARD.md`.
