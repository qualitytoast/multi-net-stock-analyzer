"""Shapley interaction index over agent coalitions (paper Eq. 22-23; plan Section 8).

v(S) = mean over evaluation days of the Spearman IC of the scheduler-reweighted score that uses
only the agents in S (weights renormalized within S); v(empty) = 0.
I(i, j) = sum over S in [n] minus {i, j} of |S|! (n - |S| - 2)! / (n - 1)! * Delta_ij v(S).
"""

from __future__ import annotations

from math import factorial

import numpy as np

from detmaso.losses.stats import daily_spearman


def coalition_values(alphas: np.ndarray, w: np.ndarray, labels: np.ndarray, label_ok: np.ndarray) -> np.ndarray:
    """v indexed by bitmask (0 .. 2^n - 1). alphas: [T, n, N], w: [T, n]."""
    n = alphas.shape[1]
    v = np.zeros(2**n)
    for mask in range(1, 2**n):
        members = [i for i in range(n) if mask >> i & 1]
        ws = w[:, members]
        ws = ws / ws.sum(axis=1, keepdims=True)
        score = np.einsum("ts,tsn->tn", ws, alphas[:, members, :])
        v[mask] = float(np.nanmean(daily_spearman(score, labels, label_ok)))
    return v


def shapley_weight(s: int, n: int) -> float:
    return factorial(s) * factorial(n - s - 2) / factorial(n - 1)


def interaction_index(v: np.ndarray, n: int) -> np.ndarray:
    """Grabisch-Roubens pairwise Shapley interaction index; NaN on the diagonal."""
    out = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            others = [k for k in range(n) if k not in (i, j)]
            total = 0.0
            for sub in range(2 ** len(others)):
                S = sum(1 << others[b] for b in range(len(others)) if sub >> b & 1)
                s = bin(S).count("1")
                delta = v[S | 1 << i | 1 << j] - v[S | 1 << i] - v[S | 1 << j] + v[S]
                total += shapley_weight(s, n) * delta
            out[i, j] = out[j, i] = total
    return out
