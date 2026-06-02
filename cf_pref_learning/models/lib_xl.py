"""D4: XL scaling of pure LIB.

Architecturally identical to LIB v0 but scaled up:
  - d_attr: 128 → 256
  - n_heads: 4 → 8
  - One additional cross-attention layer (stacked)
  - Optional FFN block per attention layer

Goal: test whether more capacity in the SAME pure-LIB design (no centroid)
breaks the LIB v0 ceiling of 0.643. No engineered centroid available;
this is a true "richer-LIB" test.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class LIBXLModule(nn.Module):
    """Scaled-up LIB: 2 attention layers, larger d_attr."""

    def __init__(
        self,
        clip_text_dim: int = 512,
        clip_patch_dim: int = 768,
        d_attr: int = 256,
        n_attr: int = 4,
        n_heads: int = 8,
        dropout: float = 0.4,
        n_layers: int = 2,
        n_color_tokens: int = 7,
        n_object_tokens: int = 7,
        n_action_tokens: int = 3,
        n_spatial_tokens: int = 3,
    ):
        super().__init__()
        assert d_attr % n_heads == 0
        self.n_attr = n_attr
        self.d_attr = d_attr
        self.q_proj = nn.Linear(clip_text_dim, n_attr * d_attr)
        self.patch_proj = nn.Linear(clip_patch_dim, d_attr)
        self.patch_ln = nn.LayerNorm(d_attr)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": nn.MultiheadAttention(d_attr, n_heads, dropout=dropout, batch_first=True),
                "ln1": nn.LayerNorm(d_attr),
                "ffn": nn.Sequential(
                    nn.Linear(d_attr, 4 * d_attr), nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(4 * d_attr, d_attr),
                ),
                "ln2": nn.LayerNorm(d_attr),
            })
            for _ in range(n_layers)
        ])
        self.expected_proj = nn.Linear(clip_text_dim, n_attr * d_attr)
        self.attr_cls = nn.ModuleDict({
            "color":   nn.Linear(d_attr, n_color_tokens),
            "object":  nn.Linear(d_attr, n_object_tokens),
            "action":  nn.Linear(d_attr, n_action_tokens),
            "spatial": nn.Linear(d_attr, n_spatial_tokens),
        })
        self._attr_names = ["color", "object", "action", "spatial"][:min(n_attr, 4)]

    def gram_offdiag_norm(self, x):
        x_n = F.normalize(x, dim=-1)
        g = x_n @ x_n.t()
        eye = torch.eye(g.shape[0], device=g.device)
        return (g - eye).abs().max()

    def forward(self, patches, text):
        B, K, P, _ = patches.shape
        Q = self.q_proj(text).view(B, self.n_attr, self.d_attr)
        P_proj = self.patch_ln(self.patch_proj(patches.view(B, K * P, -1)))
        x = Q
        for layer in self.layers:
            a, _ = layer["attn"](x, P_proj, P_proj)
            x = layer["ln1"](x + a)
            ff = layer["ffn"](x)
            x = layer["ln2"](x + ff)
        attr_emb = x  # [B, n_attr, d_attr]
        exp_emb = self.expected_proj(text).view(B, self.n_attr, self.d_attr)
        attr_n = F.normalize(attr_emb, dim=-1)
        exp_n = F.normalize(exp_emb, dim=-1)
        binding = (attr_n * exp_n).sum(dim=-1)
        recon = {axis: self.attr_cls[axis](attr_emb[:, i])
                 for i, axis in enumerate(self._attr_names)}
        return {"binding": binding, "attr_emb": attr_emb, "recon_logits": recon}


class LIBXLPreferenceHead(nn.Module):
    def __init__(self, n_attr: int = 4, clip_text_dim: int = 512,
                 text_proj_dim: int = 32, hidden: int = 128):
        super().__init__()
        self.text_proj = nn.Linear(clip_text_dim, text_proj_dim)
        self.in_dim = 3 * n_attr + text_proj_dim
        self.head = nn.Sequential(
            nn.Linear(self.in_dim, hidden), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, binding_a, binding_b, text):
        diff = binding_a - binding_b
        t = self.text_proj(text)
        feat = torch.cat([binding_a, binding_b, diff, t], dim=-1)
        return self.head(feat).squeeze(-1)


def _unit_test():
    B, K, P = 4, 8, 49
    patches = torch.randn(B, K, P, 768)
    text = torch.randn(B, 512)
    lib = LIBXLModule()
    head = LIBXLPreferenceHead()
    out = lib(patches, text)
    sc = head(out["binding"], out["binding"], text)
    assert sc.shape == (B,), sc.shape
    n = sum(p.numel() for p in lib.parameters()) + sum(p.numel() for p in head.parameters())
    print(f"LIB XL unit test PASS — {n:,} params")


if __name__ == "__main__":
    _unit_test()
