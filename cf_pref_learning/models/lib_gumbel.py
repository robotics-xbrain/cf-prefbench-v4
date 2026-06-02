"""E2: Modular routing with Gumbel-Softmax hard gating.

Hypothesis from Codex final review: soft gating allows the model to
mix LIB and centroid pathways smoothly, which lets it learn to ignore
LIB (centroid pass-through). Hard gating with Gumbel-Softmax (temperature
annealed to near-zero) forces a per-instance binary routing decision —
the model must commit to LIB or centroid per sample, which should make
it harder to ignore LIB if any sample is best routed through it.

Architecture (same as lib_hybrid.py but with Gumbel routing):
  - LIB module produces binding_a / binding_b → lib_diff
  - centroid_proj produces cent_diff from engineered 173-d pair feature
  - gate_net (instruction-conditioned MLP) produces 2-d logits
    [g_lib_logit, g_cent_logit]
  - Gumbel-Softmax samples a 2-d one-hot at training time, tau annealed
  - At inference, use argmax (true hard gating)
  - final_diff = gate[0] * lib_diff + gate[1] * cent_diff

Temperature schedule: tau(epoch) = max(0.1, 1.0 * 0.95^(epoch-1))
  epoch 1: tau ≈ 1.0 (soft)
  epoch 20: tau ≈ 0.36
  epoch 40: tau ≈ 0.13
  epoch 60+: tau = 0.1 (effectively hard)

Gate-balance regularizer: penalize if the average gate distribution
over a batch is skewed > 0.8 to one side, prevents collapse to single
pathway.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from cf_pref_learning.models.lib import LIBModule


class CentroidPairToAttrDiff(nn.Module):
    """Same as in lib_hybrid.py."""

    def __init__(self, centroid_dim: int = 173, n_attr: int = 4, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(centroid_dim, hidden), nn.GELU(),
            nn.Linear(hidden, n_attr),
        )

    def forward(self, x):
        return self.mlp(x)


class LIBGumbelSystem(nn.Module):
    """LIB + centroid + Gumbel-Softmax routing.

    The gate is 2-way per-attribute: choose LIB or centroid for each of
    n_attr attributes independently. With hard Gumbel (tau → 0.1), the
    model commits to one or the other per attribute per example.
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
            clip_text_dim=clip_text_dim, clip_patch_dim=clip_patch_dim,
            d_attr=d_attr, n_attr=n_attr, n_heads=n_heads, dropout=dropout,
        )
        self.centroid_proj = CentroidPairToAttrDiff(
            centroid_dim=centroid_dim, n_attr=n_attr
        )
        # Per-attribute 2-way gate: gate[a] = [P(use LIB on attr a), P(use centroid on attr a)]
        self.gate_net = nn.Sequential(
            nn.Linear(clip_text_dim, gate_hidden), nn.GELU(),
            nn.Linear(gate_hidden, n_attr * 2),
        )
        self.text_proj = nn.Linear(clip_text_dim, 16)
        self.in_dim = 3 * n_attr + 16
        self.head = nn.Sequential(
            nn.Linear(self.in_dim, head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )
        self.register_buffer("_tau", torch.tensor(1.0))

    def set_tau(self, tau: float):
        self._tau.fill_(float(tau))

    def forward(
        self,
        patches_a, patches_b, centroid_pair_feat, text,
        gate_override=None, hard_eval=False,
        zero_lib=False, zero_centroid=False,
    ):
        if zero_lib:
            patches_a = torch.zeros_like(patches_a)
            patches_b = torch.zeros_like(patches_b)
        out_a = self.lib(patches_a, text)
        out_b = self.lib(patches_b, text)
        lib_diff = out_a["binding"] - out_b["binding"]
        if zero_centroid:
            centroid_pair_feat = torch.zeros_like(centroid_pair_feat)
        cent_diff = self.centroid_proj(centroid_pair_feat)

        # Gate logits, reshape to [B, n_attr, 2]
        B = lib_diff.shape[0]
        g_logits = self.gate_net(text).view(B, self.n_attr, 2)

        if gate_override is not None:
            # gate_override = "lib" or "cent" or float in [0,1]
            if gate_override == "lib":
                gate = torch.zeros_like(g_logits)
                gate[:, :, 0] = 1.0
            elif gate_override == "cent":
                gate = torch.zeros_like(g_logits)
                gate[:, :, 1] = 1.0
            else:
                w = float(gate_override)
                gate = torch.zeros_like(g_logits)
                gate[:, :, 0] = w
                gate[:, :, 1] = 1 - w
        elif self.training:
            # Gumbel-Softmax sample, differentiable; tau controls hardness
            gate = F.gumbel_softmax(g_logits, tau=float(self._tau.item()), hard=False, dim=-1)
        else:
            if hard_eval:
                # Argmax routing
                idx = g_logits.argmax(dim=-1, keepdim=True)
                gate = torch.zeros_like(g_logits).scatter_(-1, idx, 1.0)
            else:
                gate = F.softmax(g_logits / float(self._tau.item()), dim=-1)

        # final_diff[a] = gate[a, 0] * lib_diff[a] + gate[a, 1] * cent_diff[a]
        final_diff = gate[:, :, 0] * lib_diff + gate[:, :, 1] * cent_diff
        t = self.text_proj(text)
        feat = torch.cat([out_a["binding"], out_b["binding"], final_diff, t], dim=-1)
        score = self.head(feat).squeeze(-1)
        return {
            "score": score, "out_a": out_a, "out_b": out_b,
            "lib_diff": lib_diff, "cent_diff": cent_diff,
            "gate": gate, "g_logits": g_logits, "final_diff": final_diff,
        }


def gate_balance_loss(gate, threshold: float = 0.8):
    """Penalize gate distribution skewed > threshold to either side.

    gate: [B, n_attr, 2]. Compute per-attribute average over batch:
    avg_gate[a] = [P(use LIB on attr a), P(use centroid on attr a)].
    Penalize if max > threshold.
    """
    avg = gate.mean(dim=0)  # [n_attr, 2]
    max_p = avg.max(dim=-1).values  # [n_attr]
    penalty = F.relu(max_p - threshold).mean()
    return penalty


def _unit_test():
    B, K, P = 4, 8, 49
    pa = torch.randn(B, K, P, 768); pb = torch.randn(B, K, P, 768)
    cf = torch.randn(B, 173); txt = torch.randn(B, 512)
    sys = LIBGumbelSystem()
    sys.train()
    out = sys(pa, pb, cf, txt)
    assert out["score"].shape == (B,)
    assert out["gate"].shape == (B, 4, 2)
    sys.eval()
    out2 = sys(pa, pb, cf, txt, hard_eval=True)
    assert out2["gate"].sum(dim=-1).allclose(torch.ones(B, 4)), "hard gate sums to 1 per attr"
    # Override
    out_lib = sys(pa, pb, cf, txt, gate_override="lib")
    out_cent = sys(pa, pb, cf, txt, gate_override="cent")
    assert torch.allclose(out_lib["final_diff"], out_lib["lib_diff"])
    assert torch.allclose(out_cent["final_diff"], out_cent["cent_diff"])
    print("LIB Gumbel unit test PASS")


if __name__ == "__main__":
    _unit_test()
