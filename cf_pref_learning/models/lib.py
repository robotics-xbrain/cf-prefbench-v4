"""Learned Instruction-conditioned Binding (LIB) module.

Implements the architecture specified in LIB_DESIGN.md §6. The
empirical motivation is the 2026-05-18 sprint finding that the
0.929 = 13/14 held-out color-binding ceiling is bound to the
engineered instruction-conditioned color-centroid pipeline, not to
any of the head-side CPL losses (8 variants × 3 seeds = 24
measurements all at 0.9286 exactly) nor to a naive frame-mean CLIP
feature (0.286) nor to a cross-attention CLIP encoder (0.190).

LIB hypothesizes that the missing inductive bias is
**instruction-conditioned discrete channel selection** over
**patch-level** visual features (not pooled), and implements that
selection differentiably via per-attribute cross-attention.

See LIB_DESIGN.md for the full spec.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class LIBModule(nn.Module):
    """Learned Instruction-conditioned Binding.

    Args:
        clip_text_dim: dimension of the CLIP text embedding (512 for ViT-B/32)
        clip_patch_dim: dimension of CLIP image-encoder patch tokens (768 for ViT-B/32)
        d_attr: per-attribute embedding dimension
        n_attr: number of attribute axes (color, object, action, spatial = 4)
        n_heads: heads in the cross-attention
        dropout: dropout rate inside the attention layer
        n_color_tokens / n_object_tokens / etc.: vocabulary sizes for the
            reconstruction auxiliary that prevents Q_attr collapse.

    Outputs (forward returns a dict):
        binding:       [B, n_attr]              — per-attribute alignment score in [-1, 1]
        attr_emb:      [B, n_attr, d_attr]      — attended attribute embeddings
        recon_logits:  dict[str, [B, n_classes]] — instruction-attribute reconstruction
    """

    def __init__(
        self,
        clip_text_dim: int = 512,
        clip_patch_dim: int = 768,
        d_attr: int = 128,
        n_attr: int = 4,
        n_heads: int = 4,
        dropout: float = 0.3,
        n_color_tokens: int = 7,
        n_object_tokens: int = 7,
        n_action_tokens: int = 3,
        n_spatial_tokens: int = 3,
    ):
        super().__init__()
        assert d_attr % n_heads == 0, "d_attr must be divisible by n_heads"
        self.n_attr = n_attr
        self.d_attr = d_attr
        # Step 1: per-attribute query projection
        self.q_proj = nn.Linear(clip_text_dim, n_attr * d_attr)
        # Step 2: patch token projection to attr dim
        self.patch_proj = nn.Linear(clip_patch_dim, d_attr)
        self.patch_ln = nn.LayerNorm(d_attr)
        # Step 3: cross-attention
        self.attn = nn.MultiheadAttention(
            d_attr, n_heads, dropout=dropout, batch_first=True
        )
        self.attn_ln = nn.LayerNorm(d_attr)
        # Step 5: expected per-attribute embedding (from instruction)
        self.expected_proj = nn.Linear(clip_text_dim, n_attr * d_attr)
        # R1 mitigation: per-axis classification head over the attended embedding
        self.attr_cls = nn.ModuleDict({
            "color":   nn.Linear(d_attr, n_color_tokens),
            "object":  nn.Linear(d_attr, n_object_tokens),
            "action":  nn.Linear(d_attr, n_action_tokens),
            "spatial": nn.Linear(d_attr, n_spatial_tokens),
        })
        # The first 4 attribute queries get the per-axis recon target;
        # extra queries (n_attr > 4) have no recon target by default.
        self._attr_names = ["color", "object", "action", "spatial"][: min(n_attr, 4)]

    def gram_offdiag_norm(self, x: torch.Tensor) -> torch.Tensor:
        """For monitoring Q_attr non-collapse.

        Args:
            x: [n_attr, dim] tensor

        Returns: max absolute off-diagonal of the cosine Gram.
        """
        x_n = F.normalize(x, dim=-1)
        g = x_n @ x_n.t()
        eye = torch.eye(g.shape[0], device=g.device)
        off = (g - eye).abs()
        return off.max()

    def forward(
        self,
        patches: torch.Tensor,
        text: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute binding scores.

        Args:
            patches: [B, K, P, clip_patch_dim] — per-frame patch tokens
                (K frames per video, P patches per frame).
            text:    [B, clip_text_dim] — CLIP text embedding of instruction.

        Returns:
            dict with keys: binding [B, n_attr], attr_emb [B, n_attr, d_attr],
                recon_logits {axis -> [B, vocab]}.
        """
        B, K, P, _ = patches.shape
        # Step 1
        Q_attr = self.q_proj(text).view(B, self.n_attr, self.d_attr)
        # Step 2 — flatten over (K, P) so the attention sees all patches across all frames
        P_proj = self.patch_ln(self.patch_proj(patches.view(B, K * P, -1)))
        # Step 3 — per-attribute attention over all flattened patches
        attended, _ = self.attn(Q_attr, P_proj, P_proj)
        # Step 4 — residual + LayerNorm
        attr_emb = self.attn_ln(attended + Q_attr)  # [B, n_attr, d_attr]
        # Step 5 — binding score per attribute via cosine similarity to
        # the instruction's expected attribute embedding
        exp_emb = self.expected_proj(text).view(B, self.n_attr, self.d_attr)
        attr_n = F.normalize(attr_emb, dim=-1)
        exp_n = F.normalize(exp_emb, dim=-1)
        binding = (attr_n * exp_n).sum(dim=-1)  # [B, n_attr]
        # R1 reconstruction logits
        recon = {
            axis: self.attr_cls[axis](attr_emb[:, i])
            for i, axis in enumerate(self._attr_names)
        }
        return {
            "binding": binding,
            "attr_emb": attr_emb,
            "recon_logits": recon,
        }


class LIBPreferenceHead(nn.Module):
    """Pairwise preference head consuming LIB binding scores.

    Mirrors `BasePreferenceHead` in `base_pref.py` but accepts the
    structured LIB output (4 binding scores per trajectory + a small
    text projection) rather than a flat 173-d centroid feature.
    """

    def __init__(
        self,
        n_attr: int = 4,
        clip_text_dim: int = 512,
        text_proj_dim: int = 16,
        hidden: int = 64,
    ):
        super().__init__()
        self.text_proj = nn.Linear(clip_text_dim, text_proj_dim)
        self.in_dim = 3 * n_attr + text_proj_dim
        self.head = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        binding_a: torch.Tensor,
        binding_b: torch.Tensor,
        text: torch.Tensor,
    ) -> torch.Tensor:
        """binding_a, binding_b: [B, n_attr]; text: [B, clip_text_dim] -> [B]"""
        diff = binding_a - binding_b
        t = self.text_proj(text)
        feat = torch.cat([binding_a, binding_b, diff, t], dim=-1)
        return self.head(feat).squeeze(-1)


def _unit_test():
    """Smoke test: forward-pass shapes."""
    B, K, P = 4, 8, 49
    patches = torch.randn(B, K, P, 768)
    text = torch.randn(B, 512)
    lib = LIBModule()
    out = lib(patches, text)
    assert out["binding"].shape == (B, 4), out["binding"].shape
    assert out["attr_emb"].shape == (B, 4, 128), out["attr_emb"].shape
    for name, sh in [("color", 7), ("object", 7), ("action", 3), ("spatial", 3)]:
        assert out["recon_logits"][name].shape == (B, sh), name
    head = LIBPreferenceHead()
    score = head(out["binding"], out["binding"], text)
    assert score.shape == (B,), score.shape
    # Gram-offdiag should be a scalar
    g = lib.gram_offdiag_norm(out["attr_emb"][0])
    assert g.shape == ()
    print("LIB unit test PASS")


if __name__ == "__main__":
    _unit_test()
