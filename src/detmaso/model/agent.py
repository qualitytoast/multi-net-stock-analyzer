"""One Det-MASO agent: a 3-hidden-layer MLP shared across stocks (paper Sec 3.2.2; plan A17)."""

from __future__ import annotations

import torch
from torch import nn

from detmaso.losses.stats import masked_std, masked_zscore


ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU}


class AgentMLP(nn.Module):
    def __init__(self, d_in: int, hidden: tuple[int, ...], activation: str = "relu"):
        super().__init__()
        layers: list[nn.Module] = []
        prev = d_in
        for width in hidden:
            layers += [nn.Linear(prev, width), ACTIVATIONS[activation]()]
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., N, D] -> raw scores [..., N] (standardized later, over the universe)."""
        return self.net(x).squeeze(-1)


def standardize_scores(raw: torch.Tensor, universe: torch.Tensor, dead_std: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Paper Eq. 2 for raw outputs [B, n, N]: z-score each output across the day's universe.

    Returns (scores, raw_std). A constant output cannot be standardized, so its score that day is 0
    (DEVIATIONS.md, entry 4). Shared by every model so the rule is identical across variants.
    """
    mask = universe.unsqueeze(1).expand_as(raw)
    alphas = masked_zscore(raw, mask, dim=-1)  # zero mean, unit variance across assets
    raw_std = masked_std(raw.detach(), mask, dim=-1)
    alphas = torch.where((raw_std < dead_std).unsqueeze(-1), torch.zeros_like(alphas), alphas)
    return alphas, raw_std
