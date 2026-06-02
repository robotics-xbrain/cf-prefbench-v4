"""D5: Multi-axis joint training with per-axis prediction heads.

Hypothesis: LIB v0 reaches 0.643 by learning a single overall-preference
signal. Adding per-axis prediction targets forces the model to disentangle
attribute-specific binding, which should improve color-axis PFA.

Architecture:
  - LIB module (same as v0): produces binding_a [B, 4] and binding_b [B, 4]
  - Main score head: same as LIB v0 (binding_a, binding_b, diff, text -> scalar)
  - 4 per-axis prediction heads, each: input = binding_a[axis] - binding_b[axis]
    output = "A is preferred on this axis" logit
  - At inference, use the MAIN score (consistent with LIB v0 protocol)

The training signal:
  - Main BCE on the overall preferred A/B label
  - 4 per-axis BCEs on per-axis labels.

For this to work I need per-axis labels. Look at the row's "axis" field
in CF-PrefBench: it's the axis on which this counterfactual flip differs.
So for a row with axis="color", the per-axis prediction target for color
is the same as the overall preferred; for other axes, the prediction
target is "neutral" (no preference).

We define:
  - if row.axis == axis_name: per_axis_label = row.preferred
  - else: per_axis_label = "neutral" (use special probability 0.5, masked
    in the loss; OR use ignore_index in F.cross_entropy)

We use a 3-way classification per axis: {A, B, neutral}, where neutral
is the answer when the pair doesn't differ on that axis.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from cf_pref_learning.models.lib import LIBModule


AXES = ["color", "object", "action", "spatial"]


class LIBMultiTaskSystem(nn.Module):
    def __init__(
        self,
        clip_text_dim: int = 512,
        clip_patch_dim: int = 768,
        d_attr: int = 128,
        n_attr: int = 4,
        n_heads: int = 4,
        dropout: float = 0.3,
        head_hidden: int = 64,
    ):
        super().__init__()
        self.lib = LIBModule(
            clip_text_dim=clip_text_dim, clip_patch_dim=clip_patch_dim,
            d_attr=d_attr, n_attr=n_attr, n_heads=n_heads, dropout=dropout,
        )
        self.text_proj = nn.Linear(clip_text_dim, 16)
        in_dim = 3 * n_attr + 16
        # Main preference head
        self.main_head = nn.Sequential(
            nn.Linear(in_dim, head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )
        # Per-axis 3-way prediction heads ({A, B, neutral})
        self.axis_heads = nn.ModuleDict({
            ax: nn.Sequential(
                nn.Linear(2, 16), nn.ReLU(),
                nn.Linear(16, 3),  # 0=A, 1=B, 2=neutral
            )
            for ax in AXES[:n_attr]
        })

    def forward(self, patches_a, patches_b, text, zero_lib=False):
        if zero_lib:
            patches_a = torch.zeros_like(patches_a)
            patches_b = torch.zeros_like(patches_b)
        out_a = self.lib(patches_a, text)
        out_b = self.lib(patches_b, text)
        binding_a = out_a["binding"]  # [B, n_attr]
        binding_b = out_b["binding"]
        lib_diff = binding_a - binding_b
        t = self.text_proj(text)
        feat = torch.cat([binding_a, binding_b, lib_diff, t], dim=-1)
        main_logit = self.main_head(feat).squeeze(-1)
        # Per-axis predictions
        axis_logits = {}
        for i, ax in enumerate(self.axis_heads.keys()):
            # Input: (binding_a[axis], binding_b[axis]) -> 3-way logits
            x = torch.stack([binding_a[:, i], binding_b[:, i]], dim=-1)
            axis_logits[ax] = self.axis_heads[ax](x)
        return {
            "score": main_logit,
            "axis_logits": axis_logits,
            "binding_a": binding_a, "binding_b": binding_b,
            "out_a": out_a, "out_b": out_b,
        }


def _unit_test():
    B, K, P = 4, 8, 49
    pa = torch.randn(B, K, P, 768); pb = torch.randn(B, K, P, 768)
    txt = torch.randn(B, 512)
    sys = LIBMultiTaskSystem()
    out = sys(pa, pb, txt)
    assert out["score"].shape == (B,)
    for ax in AXES:
        assert out["axis_logits"][ax].shape == (B, 3), ax
    print("LIB multitask unit test PASS")


if __name__ == "__main__":
    _unit_test()
