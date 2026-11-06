"""Masked cross-sectional statistics: z-scores, hard ranks and exact Spearman IC."""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import rankdata

STD_FLOOR = 1e-12


def masked_zscore(x: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Population z-score of ``x`` along ``dim`` over entries where ``mask`` is True; 0 elsewhere.

    The std is floored (not epsilon-added) so a standardized vector has exactly unit variance,
    which keeps the agent correlation matrix's diagonal exactly 1 (paper l.229).
    """
    m = mask.to(x.dtype)
    n = m.sum(dim=dim, keepdim=True).clamp_min(1.0)
    mean = (x * m).sum(dim=dim, keepdim=True) / n
    centered = (x - mean) * m
    var = (centered * centered).sum(dim=dim, keepdim=True) / n
    return centered / torch.sqrt(var.clamp_min(STD_FLOOR * STD_FLOOR))


def masked_std(x: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    m = mask.to(x.dtype)
    n = m.sum(dim=dim, keepdim=True).clamp_min(1.0)
    mean = (x * m).sum(dim=dim, keepdim=True) / n
    centered = (x - mean) * m
    return torch.sqrt((centered * centered).sum(dim=dim) / n.squeeze(dim))


def zscore_rows(x: torch.Tensor) -> torch.Tensor:
    """Population z-score of each row (no mask)."""
    centered = x - x.mean(dim=-1, keepdim=True)
    var = (centered * centered).mean(dim=-1, keepdim=True)
    return centered / torch.sqrt(var.clamp_min(STD_FLOOR * STD_FLOOR))


def hard_rank(y: np.ndarray) -> np.ndarray:
    """Ascending ranks with average ties (1..n)."""
    return rankdata(y, method="average")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Exact Spearman correlation = Pearson correlation of average-tie ranks (paper Eq. 1)."""
    ra = hard_rank(a)
    rb = hard_rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def daily_spearman(scores: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """IC_t for each row t: Spearman over entries where mask is True and both values are finite."""
    out = np.full(scores.shape[0], np.nan)
    for t in range(scores.shape[0]):
        m = mask[t] & np.isfinite(scores[t]) & np.isfinite(labels[t])
        if m.sum() >= 3:
            out[t] = spearman(scores[t, m], labels[t, m])
    return out
