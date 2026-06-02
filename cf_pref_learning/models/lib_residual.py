"""B2 — Centroid as Frozen Residual.

Hypothesis: if the centroid is a *small additive residual* and its
weights do not update, LIB must carry the main signal because there's
no learnable centroid-pass-through path.

Architecture:
  binding_diff_final = LIB_diff + alpha * detach(centroid_proj(centroid_pair_feat))
  score = head([binding_a, binding_b, binding_diff_final, text_proj])

The centroid_proj output is detached so the engineered-feature path
does NOT receive gradient. The path's LINEAR layer weights are frozen
to be small / initialized but not learnable for the prediction task.

Critically, there is NO gate. The blend is a fixed-alpha sum, with the
centroid component on a no-gradient bypass.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from cf_pref_learning.models.lib import LIBModule


class CentroidPairToAttrDiff(nn.Module):
    """Same as in lib_hybrid.py — projects 173-d centroid pair feature
    to n_attr binding diff scalars."""

    def __init__(self, centroid_dim: int = 173, n_attr: int = 4, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(centroid_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_attr),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class LIBResidualSystem(nn.Module):
    """LIB + frozen, detached centroid residual.

    Args:
        alpha: fixed scaling on the centroid residual (0.1 / 0.2 / 0.3).
        freeze_centroid_proj: if True (default), centroid_proj parameters
            do not require_grad, and its output is detached on the
            forward pass.
    """

    def __init__(
        self,
        clip_text_dim: int = 512,
        clip_patch_dim: int = 768,
        centroid_dim: int = 173,
        d_attr: int = 128,
        n_attr: int = 4,
        n_heads: int = 4,
        dropout: float = 0.3,
        alpha: float = 0.2,
        freeze_centroid_proj: bool = True,
        head_hidden: int = 64,
    ):
        super().__init__()
        self.alpha = alpha
        self.n_attr = n_attr
        self.freeze_centroid_proj = freeze_centroid_proj
        self.lib = LIBModule(
            clip_text_dim=clip_text_dim,
            clip_patch_dim=clip_patch_dim,
            d_attr=d_attr,
            n_attr=n_attr,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.centroid_proj = CentroidPairToAttrDiff(
            centroid_dim=centroid_dim, n_attr=n_attr
        )
        if freeze_centroid_proj:
            for p in self.centroid_proj.parameters():
                p.requires_grad = False
        self.text_proj = nn.Linear(clip_text_dim, 16)
        self.in_dim = 3 * n_attr + 16
        self.head = nn.Sequential(
            nn.Linear(self.in_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(
        self,
        patches_a: torch.Tensor,
        patches_b: torch.Tensor,
        centroid_pair_feat: torch.Tensor,
        text: torch.Tensor,
        zero_centroid: bool = False,
        zero_lib: bool = False,
    ) -> dict:
        if zero_lib:
            patches_a = torch.zeros_like(patches_a)
            patches_b = torch.zeros_like(patches_b)
        out_a = self.lib(patches_a, text)
        out_b = self.lib(patches_b, text)
        lib_diff = out_a["binding"] - out_b["binding"]
        if zero_centroid:
            centroid_pair_feat = torch.zeros_like(centroid_pair_feat)
        cent_raw = self.centroid_proj(centroid_pair_feat)
        if self.freeze_centroid_proj:
            cent_raw = cent_raw.detach()
        # Fixed-alpha residual blend, no gate.
        final_diff = lib_diff + self.alpha * cent_raw
        t = self.text_proj(text)
        feat = torch.cat(
            [out_a["binding"], out_b["binding"], final_diff, t], dim=-1
        )
        score = self.head(feat).squeeze(-1)
        return {
            "score": score,
            "out_a": out_a,
            "out_b": out_b,
            "lib_diff": lib_diff,
            "cent_residual": self.alpha * cent_raw,
            "final_diff": final_diff,
        }


def _unit_test():
    B, K, P = 4, 8, 49
    pa = torch.randn(B, K, P, 768)
    pb = torch.randn(B, K, P, 768)
    cf = torch.randn(B, 173)
    txt = torch.randn(B, 512)
    sys = LIBResidualSystem(alpha=0.2)
    out = sys(pa, pb, cf, txt)
    assert out["score"].shape == (B,)
    # Centroid proj should be frozen
    cent_params = list(sys.centroid_proj.parameters())
    assert all(not p.requires_grad for p in cent_params), "centroid_proj should be frozen"
    print(f"LIB residual unit test PASS (n centroid params: {sum(p.numel() for p in cent_params)})")


if __name__ == "__main__":
    _unit_test()
