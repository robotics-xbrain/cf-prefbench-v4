"""LIB + engineered-centroid gated hybrid.

Phase 3 Path B per Phase 2 W6 checkpoint user directive.

The Phase 2 W3-W4 sprint already tested naive `concat(centroid,
frame-mean CLIP)` and it FAILED (0.476 vs centroid alone 0.929) due
to feature-interference: the 2048-d CLIP block dominated the 173-d
centroid in standardized input.

This module avoids that mechanism by **learned per-attribute
routing**: an instruction-conditioned gate decides how much weight
to give the LIB binding signal vs the engineered centroid signal.

  per-attribute final binding diff[a]
      = g[a] * (lib_a[a] - lib_b[a])
      + (1 - g[a]) * centroid_pair_proj[a]

where:
  - lib_a / lib_b: LIB module output binding scores for video_a / video_b
  - centroid_pair_proj: linear projection of the engineered 173-d pair
    feature down to n_attr scalars
  - g: sigmoid(MLP(text)) ∈ (0, 1), per-attribute

The engineered 173-d pair feature is already pair-level
(`scripts/run_repaired_base_pref._pair_features` returns
concat[a-b, |a-b|, text_features]), so we feed it once per pair, not
per video.

A small entropy regularizer keeps the gate from collapsing to {0, 1}.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from cf_pref_learning.models.lib import LIBModule


class CentroidPairToAttrDiff(nn.Module):
    """Project the engineered 173-d pair feature to n_attr binding diffs."""

    def __init__(self, centroid_dim: int = 173, n_attr: int = 4, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(centroid_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_attr),
        )

    def forward(self, centroid_pair_feat: torch.Tensor) -> torch.Tensor:
        return self.mlp(centroid_pair_feat)


class LIBHybridSystem(nn.Module):
    """LIB + engineered centroid + learned per-attribute gate.

    The forward expects:
      patches_a / patches_b: [B, K, P, clip_patch_dim] (LIB input)
      centroid_pair_feat:    [B, centroid_dim] (engineered pair feature,
                              standardized externally)
      text:                  [B, clip_text_dim] (CLIP text embedding)

    Output: score, plus diagnostics (gate, lib_diff, cent_diff).
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
        gate_hidden: int = 64,
        head_hidden: int = 64,
    ):
        super().__init__()
        self.n_attr = n_attr
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
        self.gate_net = nn.Sequential(
            nn.Linear(clip_text_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, n_attr),
        )
        with torch.no_grad():
            self.gate_net[-1].bias.fill_(0.0)  # gate ~ 0.5 at init
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
        gate_override: float | None = None,
    ) -> dict:
        out_a = self.lib(patches_a, text)
        out_b = self.lib(patches_b, text)
        lib_diff = out_a["binding"] - out_b["binding"]  # [B, n_attr]

        cent_diff = self.centroid_proj(centroid_pair_feat)  # [B, n_attr]

        if gate_override is not None:
            gate = torch.full_like(lib_diff, float(gate_override))
        else:
            gate = torch.sigmoid(self.gate_net(text))  # [B, n_attr]

        final_diff = gate * lib_diff + (1.0 - gate) * cent_diff  # [B, n_attr]

        # Feed final_diff into head together with the original
        # per-side LIB bindings (which carry richer info than the diff
        # alone), and a small text projection.
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
            "cent_diff": cent_diff,
            "gate": gate,
            "final_diff": final_diff,
        }


def gate_entropy_loss(gate: torch.Tensor) -> torch.Tensor:
    """Negative entropy of the gate. Add * lambda > 0 to total loss to
    penalize gate collapse to {0, 1}.
    """
    g = gate.clamp(1e-6, 1 - 1e-6)
    ent = -(g * torch.log(g) + (1 - g) * torch.log(1 - g))
    return -ent.mean()


def _unit_test():
    B, K, P = 4, 8, 49
    patches_a = torch.randn(B, K, P, 768)
    patches_b = torch.randn(B, K, P, 768)
    centroid = torch.randn(B, 173)
    text = torch.randn(B, 512)
    sys = LIBHybridSystem()
    out = sys(patches_a, patches_b, centroid, text)
    assert out["score"].shape == (B,)
    assert out["gate"].shape == (B, 4)
    assert out["lib_diff"].shape == (B, 4)
    assert out["cent_diff"].shape == (B, 4)
    # gate=1 → final_diff == lib_diff
    out1 = sys(patches_a, patches_b, centroid, text, gate_override=1.0)
    assert torch.allclose(out1["final_diff"], out1["lib_diff"]), "gate=1 should use LIB only"
    out0 = sys(patches_a, patches_b, centroid, text, gate_override=0.0)
    assert torch.allclose(out0["final_diff"], out0["cent_diff"]), "gate=0 should use centroid only"
    ent = gate_entropy_loss(out["gate"])
    assert ent.shape == ()
    print(f"LIB hybrid unit test PASS  (entropy at init: {ent.item():.4f})")


if __name__ == "__main__":
    _unit_test()
