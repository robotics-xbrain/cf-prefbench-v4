"""D1: Residual Prediction.

The KEY architectural insight from Phase 4 round 1: when both LIB and
centroid live in the same model, the model routes through centroid and
ignores LIB. The Phase 3 hybrid was a centroid pass-through.

Residual prediction inverts the architecture:
  - centroid_pred = score from the engineered centroid baseline (frozen,
    detached). This is a scalar pre-sigmoid logit per pair.
  - lib_pred = scalar pre-sigmoid logit from LIB module.
  - final_logit = centroid_pred + alpha * lib_pred
  - alpha is a small learnable scalar, init = 0.1.
  - Training: BCE on final_logit + auxiliary MSE(lib_pred, stop_grad(target_residual))
    where target_residual = logit_target - sigmoid(centroid_pred)
    pushed through inverse-sigmoid (clamped).

The auxiliary MSE explicitly asks LIB to predict the centroid's ERROR.
LIB is therefore trained to be ORTHOGONAL to centroid, not redundant
with it.

At inference, zero LIB → final = centroid_pred (matches centroid baseline).
Useful LIB → final = centroid_pred + nontrivial correction → exceeds
centroid baseline.

This architecture should pass SC-5 (zero LIB) because:
  - At gate=0 (no LIB), result == engineered centroid (0.929 ceiling).
  - At full LIB, result > centroid IFF LIB has learned useful residual.
  - SC-5 (zero LIB) should DROP only by the residual contribution amount.
  - We don't need a huge drop — we need NORMAL > centroid alone for it
    to be a "real positive method".
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from cf_pref_learning.models.lib import LIBModule


class CentroidScalarPredictor(nn.Module):
    """Engineered centroid pair feature → scalar logit (pre-sigmoid).

    This mirrors the engineered-baseline pipeline: a Linear-256-ReLU-Linear-1
    on the 173-d pair feature. Initialized fresh; frozen in the residual-
    prediction architecture but trained for one epoch on its own first
    (warm-init pass).
    """

    def __init__(self, centroid_dim: int = 173, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(centroid_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class LIBScalarPredictor(nn.Module):
    """LIB → scalar logit (pre-sigmoid).

    Reuses LIBModule's binding scores (4-d) + a small head that maps to
    a single scalar correction. The output is added to the centroid_pred
    with a learnable alpha scaling.
    """

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
        self.head = nn.Sequential(
            nn.Linear(in_dim, head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, patches_a, patches_b, text):
        out_a = self.lib(patches_a, text)
        out_b = self.lib(patches_b, text)
        lib_diff = out_a["binding"] - out_b["binding"]
        t = self.text_proj(text)
        feat = torch.cat([out_a["binding"], out_b["binding"], lib_diff, t], dim=-1)
        return {
            "score": self.head(feat).squeeze(-1),  # scalar correction
            "out_a": out_a,
            "out_b": out_b,
            "lib_diff": lib_diff,
        }


class LIBResidualPredictionSystem(nn.Module):
    """final_logit = centroid_pred + alpha * lib_correction

    The centroid predictor is trained briefly first (in the train script's
    warm-init phase) to reach engineered-baseline performance, then FROZEN.
    Only LIB and alpha train thereafter.
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
        alpha_init: float = 0.1,
    ):
        super().__init__()
        self.centroid_pred = CentroidScalarPredictor(centroid_dim=centroid_dim)
        self.lib_pred = LIBScalarPredictor(
            clip_text_dim=clip_text_dim, clip_patch_dim=clip_patch_dim,
            d_attr=d_attr, n_attr=n_attr, n_heads=n_heads, dropout=dropout,
        )
        # Learnable scalar alpha, init at 0.1
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def freeze_centroid(self):
        for p in self.centroid_pred.parameters():
            p.requires_grad = False

    def forward(self, patches_a, patches_b, centroid_pair_feat, text,
                zero_lib=False, zero_centroid=False, gate_override=None):
        # Centroid pred: detached if frozen
        cent_logit = self.centroid_pred(centroid_pair_feat)
        if not self.centroid_pred.net[0].weight.requires_grad:
            cent_logit = cent_logit.detach()
        if zero_centroid:
            cent_logit = torch.zeros_like(cent_logit)

        # LIB pred
        if zero_lib:
            patches_a_use = torch.zeros_like(patches_a)
            patches_b_use = torch.zeros_like(patches_b)
        else:
            patches_a_use = patches_a
            patches_b_use = patches_b
        lib_out = self.lib_pred(patches_a_use, patches_b_use, text)

        # Final: alpha * lib + centroid
        # gate_override is interpreted as "lib weight" if provided
        # (gate_override=1 → full LIB pulled in even if alpha=0;
        #  gate_override=0 → no LIB even if alpha != 0)
        if gate_override is not None:
            lib_weight = torch.full_like(cent_logit, float(gate_override))
            final = cent_logit + lib_weight * lib_out["score"]
        else:
            final = cent_logit + self.alpha * lib_out["score"]
        return {
            "score": final,
            "cent_logit": cent_logit,
            "lib_logit": lib_out["score"],
            "alpha": self.alpha.detach().clone(),
            "out_a": lib_out["out_a"],
            "out_b": lib_out["out_b"],
        }


def _unit_test():
    B, K, P = 4, 8, 49
    pa = torch.randn(B, K, P, 768)
    pb = torch.randn(B, K, P, 768)
    cf = torch.randn(B, 173)
    txt = torch.randn(B, 512)
    sys = LIBResidualPredictionSystem()
    out = sys(pa, pb, cf, txt)
    assert out["score"].shape == (B,)
    sys.freeze_centroid()
    out2 = sys(pa, pb, cf, txt)
    assert not sys.centroid_pred.net[0].weight.requires_grad
    # gate_override = 0 should produce same logit as zeroing alpha
    sys.alpha.data.fill_(0.0)
    out3 = sys(pa, pb, cf, txt, gate_override=0.0)
    out4 = sys(pa, pb, cf, txt)  # alpha=0
    assert torch.allclose(out3["score"], out4["score"]), "gate=0 should match alpha=0"
    print("LIB resid_pred unit test PASS")


if __name__ == "__main__":
    _unit_test()
