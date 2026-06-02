from __future__ import annotations

import torch
import torch.nn.functional as F


def counterfactual_reversal_loss(score_pos: torch.Tensor, score_cf: torch.Tensor, margin: float = 0.1, lambda_margin: float = 0.0) -> torch.Tensor:
    ce = F.binary_cross_entropy_with_logits(score_pos - score_cf, torch.ones_like(score_pos))
    margin_loss = F.relu(margin - (score_pos - score_cf)).mean()
    return ce + lambda_margin * margin_loss

