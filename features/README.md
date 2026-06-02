# features/

Small feature caches for the smoke test and cross-family text side:
- `smoke_test_features.npz` — tiny purpose-built cache (video_patches [8,8,196,768],
  text_features [18,768]) used by `scripts/smoke_test.sh` for a real dry-run forward.
- `openclip_laion2b_text_features.npz` — OpenCLIP-LAION-2B text features (3240×512).

Full frozen-CLIP feature caches (831 MB B/32, 3.9 GB L/14, 2.3 GB SigLIP, ~1.7 GB OOD)
are EXCLUDED for size; regenerate with `scripts/eval/extract_*`.
