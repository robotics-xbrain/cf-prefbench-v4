from __future__ import annotations

import torch
from torch import nn


class BasePreferenceHead(nn.Module):
    """Pairwise preference head for precomputed allowed-input features.

    The caller is responsible for supplying features derived only from decoded
    video frames and raw instruction text. Structured benchmark fields such as
    axis, metadata, ids, split names, and filename tokens are not valid inputs.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)
