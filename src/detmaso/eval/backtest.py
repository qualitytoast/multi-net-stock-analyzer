"""Top-decile long-only backtest, rebalanced daily (plan A36-A40, D3 = a).

At the close of day t: from the eligible stocks (universe_t and traded on t), hold the top
ceil(10%) by alpha_hat_t in equal weights until the close of t+1. A held stock with no price on
t+1 earns 0 that day (A38). No costs or price-limit rules (A39).
"""

from __future__ import annotations

import math

import numpy as np


def top_decile_backtest(alpha_hat: np.ndarray, tradable: np.ndarray, r_next: np.ndarray, top_frac: float = 0.10):
    """Returns (portfolio returns [T], equal-weighted eligible-universe returns [T], picks list)."""
    T = alpha_hat.shape[0]
    port = np.full(T, np.nan)
    bench = np.full(T, np.nan)
    picks: list[np.ndarray] = []
    for t in range(T):
        elig = np.flatnonzero(tradable[t] & np.isfinite(alpha_hat[t]))
        if len(elig) == 0:
            picks.append(elig)
            continue
        k = math.ceil(top_frac * len(elig))
        order = elig[np.argsort(-alpha_hat[t, elig], kind="stable")]
        chosen = order[:k]
        picks.append(chosen)
        port[t] = np.nan_to_num(r_next[t, chosen], nan=0.0).mean()
        bench[t] = np.nan_to_num(r_next[t, elig], nan=0.0).mean()
    return port, bench, picks
