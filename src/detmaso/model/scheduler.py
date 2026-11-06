"""Context-conditioned scheduler (paper Eq. 6-8; plan A19).

s_t (12 dims, grouped by window as [sigma, mu, delta] x {5, 20, 60, 120}) -> per-window two-layer
MLP encoders (R^3 -> R^16) -> attention over the 4 windows (W_a in R^{4x64}, no bias, as in Eq. 6)
-> fused h_t in R^16 -> W_2 ReLU(W_1 h_t + b_1) + b_2 -> softmax weights over the n agents.
"""

from __future__ import annotations

import torch
from torch import nn


class Scheduler(nn.Module):
    def __init__(self, n_agents: int, n_windows: int = 4, stats_per_window: int = 3, enc_dim: int = 16, head_hidden: int = 32):
        super().__init__()
        self.n_windows = n_windows
        self.stats_per_window = stats_per_window
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(stats_per_window, enc_dim), nn.ReLU(), nn.Linear(enc_dim, enc_dim))
                for _ in range(n_windows)
            ]
        )
        self.attn = nn.Linear(n_windows * enc_dim, n_windows, bias=False)  # W_a
        self.head1 = nn.Linear(enc_dim, head_hidden)  # W_1, b_1
        self.head2 = nn.Linear(head_hidden, n_agents)  # W_2, b_2
        # "The scheduler's output layer is initialized to zero so that w_0 is uniform" (l.348).
        nn.init.zeros_(self.head2.weight)
        nn.init.zeros_(self.head2.bias)

    def forward(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """s: [B, 12] -> (agent logits [B, n], window attention a_t [B, 4])."""
        parts = s.view(s.shape[0], self.n_windows, self.stats_per_window)
        h = torch.stack([enc(parts[:, k]) for k, enc in enumerate(self.encoders)], dim=1)  # [B, 4, 16]
        a = torch.softmax(self.attn(h.flatten(1)), dim=-1)  # [B, 4]
        fused = (a.unsqueeze(-1) * h).sum(dim=1)  # [B, 16]
        logits = self.head2(torch.relu(self.head1(fused)))  # [B, n]
        return logits, a
