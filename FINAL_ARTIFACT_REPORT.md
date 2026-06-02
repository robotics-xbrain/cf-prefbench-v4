# FINAL ARTIFACT REPORT — CF-PrefBench v4 / LIB v0 anonymous artifact

## 1–5. Location, package, size

| Field | Value |
|---|---|
| Artifact directory | `/path/to/cf-prefbench-v4-anonymous` (built at `/data3/.../cf-prefbench-v4-anonymous`) |
| Zip path | `/path/to/cf-prefbench-v4-anonymous.zip` |
| Zip SHA256 | `8554e665534d4bd9f4a97c85baa3372f7c1d1055017a300ae44c715453fc34cc` (also in `cf-prefbench-v4-anonymous.zip.sha256` sidecar) |
| Zip size | 148,828,409 bytes (≈ 142 MB) |
| Zip file count | 417 entries (364 files + directories) |
| Unpacked size | ≈ 178 MB |
| File count (unpacked) | 366 |
| Source PDF SHA256 | `40901f289af2bdcab9abe42511f99b957f3ecaabeb2c811c634a1f886c6e37c8` (16 pp) |

## 6. Data included
CF-PrefBench v4 (8 core splits, 7,749 rows; 7 axes + impossible_premise) + 16 expanded
per-token size/speed test splits + scoot probe; CF-PrefBench v3 (4-axis, Appendix G);
generation-summary provenance (Tables 9–10); 6 sample videos. Per-split counts verified
== Table 9 (train 528/axis, val/held-lex/color/spatial 84, seen/camera/hardneg 72,
impossible_premise 27/non-train; total 7,749).

## 7. Checkpoints included (48 `*_best.pt`, 137 MB)
B/32 main 8-seed (= B-B), L/14 main 8-seed (= L-L), factorial B-L/L-B, cross-family
OC-OC/SG-SG, alt heads (bilinear/MLP/crossattn/linear/CLIPScore), substitution, sanity.
Full map in `docs/CHECKPOINTS.md`. Third-party encoder weights NOT included.

## 8. Scripts included
Training (`train_lib*.py`), feature extraction (`extract_*`), eval (`evaluate_lib_*`),
probes (substitution / SC-1..6 sanity / VLM judges), data generation, figure & table
makers, and artifact scripts (smoke/verify/reproduce). 37 `.py` + 4 `.sh`.

## 9. Result files included
`MASTER_PAPER_DATA.tex` (single source of truth) + per-seed `class_aggregates` metrics
(EXP-B/D/F/G/H) + aggregate cliff JSONs + VLM (Table 19) + 7-axis/v3 (Tables 11–13) +
sanity (Tables 15–17, App J) + expanded-token n=8 pool (Tables 5/7).

## 10. Figures included
`figures/main` Fig 2,3,4; `figures/appendix` Fig 5,6 + 3 supplementary; generators in
`scripts/figures`; `figures/reproduced` re-rendered Fig 2–4. Mapping in `figures/README.md`.
Figure 1 (architecture) is hand-drawn → `docs/EMNLP_final.pdf` p.3.

## 11. Mapping to final PDF
Every detected item (Tables 1–19, Figures 1–6, Appendices A–M) has a row in
`docs/PAPER_EVIDENCE_MAP.md`. Main items (Tables 1–5, Figs 1–4) and Appendix items
(Tables 6–19, Figs 5–6) all have INCLUDE-status evidence.

## 12. Excluded (with reason)
- Third-party CLIP/OpenCLIP/SigLIP weights → download at runtime.
- Multi-GB frozen-CLIP feature caches (831 MB B/32, 3.9 GB L/14, 2.3 GB SigLIP, ~1.7 GB
  OOD) and intermediate `ckpt_epoch_*.pt` (~400 MB) → regenerate via scripts.
- Full raw videos (62 MB) → 6 samples shipped; regenerate via generator.
- Planning notes, weekly/farewell reports, prompts, review-stage, research-wiki, logs,
  VCS/caches → not part of the final PDF (see `docs/REPO_FILE_CLASSIFICATION.md`).

## 13. Optional candidates (not shipped)
EXP-H **B-OC** misaligned cross-family cell (not reported in final §4.4); hybrid
checkpoint for Table 15 (numbers preserved in results). Easy to add if requested.

## 14. Missing items
- **Appendix M (CLEVRER / ManiSkill OOD):** per-trial prediction JSONs and the OOD
  speed-cliff eval scripts were **not recoverable** as discrete files; oversized feature
  caches (~1.7 GB) excluded. Reported numbers preserved in `docs/EMNLP_final.pdf` +
  `results/MASTER_PAPER_DATA.tex`. → **WARNING** (single, non-main item).

## 15. Identity scan — PASS
0 high-risk hits in text, 0 in `.pt`/`.npz`/`.mp4` binaries, 0 in the PDF
(author block already `Anonymous ACL submission`). Stray `run.log` + `__pycache__`
removed. Details: `docs/IDENTITY_LEAK_SCAN.md`. No blocking anonymity issues.

## 16. Smoke test — PASS
`smoke_test_results.json`: environment PASS (py3.10, numpy 2.0.1, torch 2.5.1);
dataset_rows PASS (total 7,749); feature_cache PASS; checkpoint PASS (703,013 params
loaded from `lib_v0_b32_seed1_best.pt`); **dry_run_forward PASS** (real LIB forward
produced preference scores). Genuinely executed — not fabricated.

## 17. verify_artifact — PASS
See `verify_artifact_results.json` (structure, key assets, hygiene: no .git/__pycache__/
logs/wandb, identity scan + smoke result present).

## 18. Upload readiness
Buildable, runnable, anonymized artifact ready for `anonymous.4open.science`
**pending the human checks in §19**.

## 19. Human checks before upload
1. Open `docs/EMNLP_final.pdf` and confirm author block reads `Anonymous ACL submission`.
2. Spot-check a few `data/*.jsonl` rows and one checkpoint load.
3. Decide whether to include the B-OC optional cell / hybrid checkpoint.
4. Confirm Appendix M WARNING is acceptable (OOD per-trial artifacts unavailable).
5. Confirm 4open.science size limits accommodate ~178 MB unpacked (zip ≈ 142 MB).
6. Note: the `FINAL_ARTIFACT_REPORT.md` copy *inside* the zip shows placeholders for
   the zip's own hash (a file cannot contain its archive's checksum); the authoritative
   zip SHA256 is in the on-disk report above and the `.zip.sha256` sidecar.

## 20. FINAL STATUS
**READY FOR UPLOAD (with 1 WARNING: Appendix M OOD per-trial artifacts not included; reported numbers preserved in the PDF).**
All main-paper tables/figures have evidence; identity scan, smoke test, and
verify_artifact all PASS. Recommend the §19 human spot-check before publishing.
