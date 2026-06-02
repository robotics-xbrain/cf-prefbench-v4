# Model card — LIB v0 (Learned Instruction-conditioned Binding)

## Architecture
A per-attribute cross-attention head over **frozen** CLIP features (Figure 1,
Appendix H). Code: `cf_pref_learning/models/lib.py`.

1. A linear projection maps the CLIP text embedding to `n=4` attribute query vectors
   (`d_attr=128` each).
2. Queries attend over the flattened `K·P` patch tokens (`K=8` sampled frames;
   `P=49` for B/32, `P=256` for L/14) via a 4-head attention + LayerNorm + residual.
3. A second projection produces 4 "expected" per-attribute embeddings; cosine
   similarity (attended vs expected) gives a 4-d binding score in `[-1,1]` per video.
4. The 4-d binding vector + a 16-d text projection are concatenated across the two
   videos and passed through a 64-hidden-unit MLP → scalar preference score
   (`LIBPreferenceHead`).
5. A reconstruction auxiliary (per-axis classifier over attended embeddings) prevents
   attribute-query collapse.

**Trainable params:** ~0.6M (B/32) / ~0.9M (L/14). Frozen CLIP visual encoder:
~87M (B/32) / ~305M (L/14). Cross-family / B-L cells add a learnable linear
pre-projection to a fixed 512 before LIB (e.g. SigLIP 768→512).

## Alternative heads (Appendix D / Figure 6)
`cf_pref_learning/models/lib_*.py` + `experiments/EXP-D`: bilinear `tᵀWv`,
2-layer MLP over `[t‖v̄]`, cross-attention without attribute queries; plus two
omitted-baseline degenerate heads (linear probe, zero-shot CLIPScore).

## Training
AdamW, 60 epochs, dropout 0.3, batch 32 (16 for L/14). Losses: preference BCE +
reconstruction CE + counterfactual margin + paraphrase stability (1.0/0.1/0.05/0.02).
lr_LIB 1e-4, lr_head 5e-4, lr_preproj 1e-4, weight-decay 1e-4. Seeds: {1,2,3}
canonical; B/32 & L/14 also have an 8-seed pool {1,2,3,7,31,99,256,2025}; cross-family
and factorial off-diagonal cells use {42,123,2024}.

## Inputs / outputs
Input: precomputed CLIP patch features `[B,K,P,patch_dim]` + CLIP text embedding
`[B,text_dim]`. Output: per-attribute binding scores → pairwise preference score.

## Intended use & limitations
A **diagnostic** preference model, not a SOTA target. The paper's point is that such
small heads over frozen CLIP text exhibit a lexical cliff. Labels are simulator-derived.
The cliff is established on B/32, partially closed on L/14, and is encoder-family
dependent (see paper Limitations). Do not deploy as a real-world preference model.

## Checkpoints
48 `*_best.pt` files; see `docs/CHECKPOINTS.md` for the full checkpoint→table/figure→
seed mapping.
