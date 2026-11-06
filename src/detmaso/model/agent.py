"""One Det-MASO agent: a 3-hidden-layer MLP shared across stocks (paper Sec 3.2.2; plan A17)."""

from __future__ import annotations

import torch
from torch import nn


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
