# PAPER ITEM INDEX — CF-PrefBench v4 / LIB v0 anonymous artifact

> **Source of truth.** This index is built **only** from the final submission PDF.
> The Overleaf LaTeX source is **not available**; the PDF (parsed with `pdftotext`)
> is therefore the sole authority for which tables, figures, and appendix results
> the artifact must cover. Older `paper/`, `paper_updates/`, and draft PDFs were
> consulted only to *locate* code/result/figure files and never to decide scope.

## Selected final paper PDF

| Field | Value |
|---|---|
| PDF path | `/path/to/project/final_paper_snapshot/overleaf_final/EMNLP_final_overleaf.pdf` |
| SHA256 | `40901f289af2bdcab9abe42511f99b957f3ecaabeb2c811c634a1f886c6e37c8` |
| File size | 1,029,235 bytes (≈ 1.00 MB) |
| Page count | 16 (A4, 595.276 × 841.89 pts) |
| Extracted text | `outputs/anonymous_artifact_build/EMNLP_final_overleaf.txt` (3,365 lines) |

**Title:** *When Paraphrases Break Binding: A Lexical Cliff in Small Heads over Frozen CLIP Text Embeddings*

**Author block:** `Anonymous ACL submission` (already anonymized in the PDF).

**Abstract (first lines):** "Small learned heads over frozen CLIP text embeddings are routinely evaluated on held-out examples whose lexical cue variants still come from the training pool, leaving lexical generalization untested. We study this gap on a controlled compositional preference task and identify a *lexical cliff*: a per-attribute cross-attention head over frozen ViT-B/32 features loses accuracy on held-out cue variants even when they sit inside a narrow CLIP-text neighbourhood (cosine [0.87, 0.97] to training) and the visual input is unchanged. …"

**Artifact release statement (verbatim):**
- Abstract / §3: "We release CF-PrefBench v4 and recommend lexical-held-out splits with cosine stratification for evaluating small learned heads over frozen CLIP text."
- §7 Conclusion: "**CF-PrefBench v4, LIB v0 checkpoints, and probe scripts are released as an anonymous artifact.**"
- Contributions (3): "CF-PrefBench v4 (7 binding axes, 7,749 rows, cosine-stratified held-out cue variants), released as an anonymous artifact along with auxiliary diagnostic scripts (§5)…"
- §5 / Ethics: "The probes are released alongside the artifact … by releasing the full probe-runner scripts."
- Appendix M: "All scripts, 5 QC mp4 visualisations … and per-trial predictions are released alongside the artifact."

Therefore the artifact scope = **CF-PrefBench v4 data + LIB v0 model/checkpoints + probe & diagnostic scripts + the result/figure sources behind the tables/figures/appendices below.**

---

## Main-paper figures (detected in PDF)

| Fig | Caption (abridged) | PDF loc |
|---|---|---|
| Figure 1 | LIB v0 architecture (frozen CLIP encoders → 4-head cross-attention + cosine binding score + MLP; reconstruction auxiliary) | §3, p.3 |
| Figure 2 | Cliff-accuracy heatmap under the 2×2 {visual}×{text} encoder factorial | §4.3, p.5 |
| Figure 3 | Cross-family validation of the lexical cliff (B-B / OC-OC / SG-SG, six cliff classes) | §4.4, p.6 |
| Figure 4 | Two cliff mechanisms: (a) motion cosine-correlated decline (n=12, r=0.627); (b) size/speed class-bimodal defaulting | §4.5, p.6 |
| Figure 5 | Motion-verb cliff under two backbones (B/32 vs L/14 across 4 verbs) | App. C, p.11 |
| Figure 6 | Cliff reproduces on 3/3 architecturally viable learned heads over frozen CLIP-B/32 | App. D, p.11 |

## Main-paper / appendix tables (detected in PDF)

| Tbl | Caption (abridged) | Section |
|---|---|---|
| Table 1 | ViT-B/32 motion-verb cliff on the 4-verb cliff probe (shift/convey/scoot/transit) | §4.1 |
| Table 2 | Motion-verb cliff under two backbones (B/32 vs L/14) | §4.2 |
| Table 3 | Cliff accuracy under the 2×2 {visual}×{text} encoder factorial | §4.3 |
| Table 4 | Cross-family validation (matched n=3): B-B, OC-OC (LAION-2B), SG-SG (SigLIP) | §4.4 |
| Table 5 | Three-axis × two-architecture cliff matrix on the expanded held-out token pool (n=8 seeds) | §4.5 |
| Table 6 | Substitution recovery on ViT-B/32 motion verbs | App. A |
| Table 7 | Per-token cliff results on the expanded n=16 size/speed pool + 4 motion (n=8 seeds, B/32 & L/14) | App. C |
| Table 8 | Numerical values for Figure 6 (above-vs-below gap, bilinear/MLP/X-attn, 3 seeds) | App. D |
| Table 9 | Per-split sizes (identical across all 7 binding axes) + impossible_premise | App. E |
| Table 10 | Train and held-out paraphrase pools per binding axis (disjoint) | App. E |
| Table 11 | LIB-B/32 held-out-lexical accuracy across all 7 binding axes (cliff incidence) | App. F |
| Table 12 | v3 action axis: test_seen vs test_heldout_lexical row accuracy, 3 seeds | App. G |
| Table 13 | v3 action axis: per-held-out-instruction row accuracy (3-seed mean) | App. G |
| Table 14 | Probe applicability per method class (TF = trivial-fail) | App. I.2 |
| Table 15 | Phase 3 hybrid sanity audit, v3 test_heldout_color (SC-1…SC-6) | App. I.3 |
| Table 16 | Pure-LIB sanity probes on the motion_sequence axis (SC-6 zero/shuffle, substitution) | App. I.3 |
| Table 17 | Phase 4 anti-collapse sanity matrix (4 variants, none passes all targets) | App. I.3 |
| Table 18 | Per-token cliff accuracy under three contrastive systems (cross-family per-token) | App. K |
| Table 19 | Per-class zero-shot VLM accuracy on the cliff test set (GPT-4o, Qwen2.5-VL) | App. L |

## Appendix sections (detected in PDF)

| App | Title | Backs |
|---|---|---|
| A | Substitution Recovery on Motion Verbs | Table 6 |
| B | Expanded Motion-Verb Pool (n=12) + confound controls | Figure 4(a), §4.1 correlation |
| C | Per-Token Cliff Results and Cliff Figures | Table 7, Figure 5 |
| D | Robustness Across Alternative Heads (full results) | Figure 6, Table 8 |
| E | CF-PrefBench v4 Split Sizes and Paraphrase Pools | Tables 9, 10 |
| F | Cliff Incidence Across All Seven Axes | Table 11 |
| G | Replication on the v3 Action Axis | Tables 12, 13 |
| H | LIB v0 Architecture Details | Figure 1 |
| I | Sanity Battery Details (I.1 defs, I.2 applicability, I.3 raw data) | Tables 14, 15, 16, 17 |
| J | Class-Prior Audit (uninformative-text probe) | §6 label-balance claim |
| K | SigLIP Cross-Family Per-Token Results | Table 18 |
| L | Zero-Shot VLM Decodability of Cliff Tokens | Table 19 |
| M | Out-of-Domain Rendered Validation (CLEVRER & ManiSkill PushCube) | §4.7 generalization probes |

---

## Cross-check against the task's expected checklist

The task supplied an expected checklist. Reconciliation:

- **All Table 1–19 present** in the final PDF (verified). The task's Table-8/Table-15 ordering note matches the PDF (float placement swaps Table 8↔9 and Table 14↔15 visually; numbering is as listed above).
- **All Figures 1–6 present.**
- **Appendices A–M all present.** Appendix J in the PDF is titled "Class-Prior Audit" (contains the uninformative-text probe the task expected under "label balance / uninformative-text probe").
- No detected table/figure/appendix item is *absent* from the selected PDF. Nothing extra needed to be added beyond 1–19 / A–M.

## Unresolved / ambiguous items (for the evidence map)

1. **Appendix M (CLEVRER & ManiSkill OOD).** The PDF reports numeric results and states "all scripts, 5 QC mp4 visualisations, and per-trial predictions are released." In the repo only **oversized feature caches** (`realdata_validation/{features,maniskill/features}`, ~1.7 GB) and **qualitative robot videos** were located; the discrete OOD speed-cliff **eval scripts and per-trial prediction JSONs were not found** as standalone files. → flagged `MISSING_SOURCE_DATA` / WARNING in PAPER_EVIDENCE_MAP; numbers are preserved in `MASTER_PAPER_DATA.tex` and the PDF.
2. **Figure 1 (architecture).** Hand-drawn (TikZ in Overleaf); no standalone vector source exists in the repo. The final PDF (page 3) is the authoritative rendering; included as `docs/EMNLP_final.pdf`.
3. **Camera-ready figure file ↔ final-PDF figure-number mapping.** Camera-ready filenames use the *older* section numbering (`fig_4_1`…`fig_4_5`); the final PDF reorganized figure order. Mapping resolved by content and documented in `figures/README.md`.
4. **Protocol-dependent numbers.** Several tables report the same cells under different seed/token protocols (n=3 vs n=8; seeds {1,2,3} vs {42,123,2024}). Each table caption specifies its protocol; the artifact ships both the n=8 canonical pool (`expanded_tokens/`) and the per-experiment n=3 cells (`experiments/EXP-*`).

## Note

PDF is the only source of truth because the Overleaf LaTeX source is not available. All scope decisions above derive from the parsed text of this single PDF.
