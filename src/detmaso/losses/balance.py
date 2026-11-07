"""Scheduler load-balancing penalty (TODO "Fix the scheduler collapse"; rules in docs/RULES.md).

usage_i = the scheduler's average weight on agent i over all training days. The scheduler sees only each
day's 12 market-state inputs, so this is one small batched forward pass (about 1 ms per step). Standard
mixture-of-experts balance losses measure usage over a batch of inputs; here the batch is one day, so the
usage is measured exactly over every training day instead (no lag, no dependence on the day order).

penalty = KL(target || usage) = sum_i target_i * ln(target_i / usage_i). The log form keeps pushing on a
starved agent: relative to the other loss terms, whose pull on an agent shrinks with its weight, the
penalty's push grows as the agent starves (the quadratic form's push stays a fixed multiple of c_bal).
The penalty acts on usage over time, not on each day's weights, so a day can still pick one agent sharply;
it would equally accept a constant blend, which is why the rules also check routing.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

USAGE_FLOOR = 1e-300  # usage below this is treated as this (keeps ln finite in float64)


def target_shares(kind: str, regimes, n_agents: int) -> torch.Tensor:
    """Target usage per agent (float64, sums to 1): equal shares, or each regime's share of the given days."""
    if kind == "uniform":
        return torch.full((n_agents,), 1.0 / n_agents, dtype=torch.float64)
    if kind == "regime":
        counts = np.bincount(np.asarray(regimes, dtype=np.int64).ravel(), minlength=n_agents)
        if len(counts) != n_agents:
            raise ValueError(f"regime labels go up to {len(counts) - 1}, but there are {n_agents} agents")
        if counts.sum() == 0:
            raise ValueError("no days to compute regime shares from")
        return torch.as_tensor(counts / counts.sum(), dtype=torch.float64)
    raise ValueError(f"unknown balance target {kind!r}")


def kl_to_target(target: torch.Tensor, usage: torch.Tensor) -> torch.Tensor:
    """KL(target || usage), with 0 * ln 0 = 0 (agents with a zero target are ignored) and usage floored."""
    pos = target > 0
    t = target[pos].to(torch.float64)
    u = usage[pos].to(torch.float64).clamp_min(USAGE_FLOOR)
    return (t * (torch.log(t) - torch.log(u))).sum()


def scheduler_usage(scheduler: nn.Module, s: torch.Tensor) -> torch.Tensor:
    """Mean softmax weight per agent over the rows of s ([days, 12]); float64."""
    logits, _ = scheduler(s)
    return torch.softmax(logits.double(), dim=-1).mean(dim=0)


def effective_agents(usage) -> float:
    """e^(entropy of the usage): 1 = one agent does everything, n = all used equally."""
    u = np.asarray(usage, dtype=np.float64)
    u = u[u > 0]
    return float(math.exp(-(u * np.log(u)).sum()))


class BalancePenalty:
    """KL(target || usage of the scheduler over the training days), differentiable in the scheduler."""

    def __init__(self, s_train: torch.Tensor, target: torch.Tensor):
        if s_train.ndim != 2:
            raise ValueError("s_train must be [days, state dims]")
        self.s_train = s_train
        self.target = target.to(device=s_train.device, dtype=torch.float64)

    def __call__(self, scheduler: nn.Module) -> torch.Tensor:
        return kl_to_target(self.target, scheduler_usage(scheduler, self.s_train))

    @torch.no_grad()
    def usage(self, scheduler: nn.Module) -> np.ndarray:
        return scheduler_usage(scheduler, self.s_train).cpu().numpy()
