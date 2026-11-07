"""Step 1 baseline (TODO.md): one plain network instead of Det-MASO's agents + scheduler.

Shaped like one Det-MASO agent (model.hidden) and standardized exactly like one (paper Eq. 2,
DEVIATIONS 4), so its score is directly comparable. There is no scheduler (weight 1), no regime
mask and no diversity or entropy term; the trainer gives it the IC and Sharpe terms only. With
model.market_state_input, every stock's features are extended with the day's 12 market-state
inputs (the scheduler's s_t, built from data through t-1), i.e. the information the scheduler sees.
"""

from __future__ import annotations

import torch
from torch import nn

from detmaso.config import Config
from detmaso.model.agent import AgentMLP, standardize_scores
from detmaso.model.detmaso import Output


class SingleMLP(nn.Module):
    n_agents = 1
    uses_regimes = False

    def __init__(
        self,
        d_in: int,
        hidden: tuple[int, ...],
        activation: str = "gelu",
        dead_std: float = 1e-12,
        market_state_input: bool = False,
        n_state: int = 12,
    ):
        super().__init__()
        self.dead_std = dead_std
        self.market_state_input = market_state_input
        self.agent = AgentMLP(d_in + (n_state if market_state_input else 0), hidden, activation)

    @classmethod
    def from_config(cls, cfg: Config, d_in: int) -> "SingleMLP":
        return cls(
            d_in=d_in,
            hidden=cfg.model.hidden,
            activation=cfg.model.activation,
            dead_std=cfg.train.dead_std,
            market_state_input=cfg.model.market_state_input,
            n_state=3 * len(cfg.regimes.windows),  # (volatility, momentum, dispersion) per window
        )

    def forward(self, X: torch.Tensor, universe: torch.Tensor, s: torch.Tensor) -> Output:
        """X: [B, N, D], universe: [B, N] bool, s: [B, 12] market state (used only if enabled)."""
        if self.market_state_input:
            X = torch.cat([X, s.unsqueeze(1).expand(-1, X.shape[1], -1)], dim=-1)
        raw = self.agent(X).unsqueeze(1)  # [B, 1, N]
        alphas, raw_std = standardize_scores(raw, universe, self.dead_std)
        batch = X.shape[0]
        return Output(
            alphas=alphas,
            alpha_hat=alphas[:, 0],
            logits=alphas.new_zeros(batch, 1),
            w=alphas.new_ones(batch, 1),
            a=alphas.new_zeros(batch, 0),  # no window attention
            raw_std=raw_std,
        )
