"""Det-MASO: n independent agents + context-conditioned scheduler (paper Eq. 2, 9)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from detmaso.config import Config
from detmaso.losses.stats import masked_std, masked_zscore
from detmaso.model.agent import AgentMLP
from detmaso.model.scheduler import Scheduler


@dataclass
class Output:
    alphas: torch.Tensor  # [B, n, N] per-agent scores, z-scored over the universe (0 outside)
    alpha_hat: torch.Tensor  # [B, N] final score = sum_i w_i alpha_i
    logits: torch.Tensor  # [B, n] scheduler logits
    w: torch.Tensor  # [B, n] agent weights (softmax)
    a: torch.Tensor  # [B, 4] window attention
    raw_std: torch.Tensor  # [B, n] std of raw agent outputs over the universe (dead-agent check)


class DetMASO(nn.Module):
    def __init__(
        self,
        n_agents: int,
        d_in: int,
        hidden: tuple[int, ...],
        enc_dim: int = 16,
        head_hidden: int = 32,
        activation: str = "relu",
        dead_std: float = 1e-12,
    ):
        super().__init__()
        self.dead_std = dead_std
        # Independent parameters per agent (no sharing), created in order from the seeded RNG.
        self.agents = nn.ModuleList([AgentMLP(d_in, hidden, activation) for _ in range(n_agents)])
        self.scheduler = Scheduler(n_agents, enc_dim=enc_dim, head_hidden=head_hidden)

    @classmethod
    def from_config(cls, cfg: Config, d_in: int) -> "DetMASO":
        return cls(
            n_agents=cfg.regimes.n_agents,
            d_in=d_in,
            hidden=cfg.model.hidden,
            enc_dim=cfg.model.encoder_hidden,
            head_hidden=cfg.model.head_hidden,
            activation=cfg.model.activation,
            dead_std=cfg.train.dead_std,
        )

    def forward(self, X: torch.Tensor, universe: torch.Tensor, s: torch.Tensor) -> Output:
        """X: [B, N, D], universe: [B, N] bool, s: [B, 12] scheduler input."""
        raw = torch.stack([agent(X) for agent in self.agents], dim=1)  # [B, n, N]
        mask = universe.unsqueeze(1).expand_as(raw)
        alphas = masked_zscore(raw, mask, dim=-1)  # Eq. 2: zero mean, unit variance across assets
        raw_std = masked_std(raw.detach(), mask, dim=-1)
        # A constant output cannot be standardized; its score that day is 0 (DEVIATIONS.md, entry 4).
        alphas = torch.where((raw_std < self.dead_std).unsqueeze(-1), torch.zeros_like(alphas), alphas)
        logits, a = self.scheduler(s)
        w = torch.softmax(logits, dim=-1)
        alpha_hat = (w.unsqueeze(-1) * alphas).sum(dim=1)  # Eq. 9
        return Output(alphas, alpha_hat, logits, w, a, raw_std)
