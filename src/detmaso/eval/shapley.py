"""Shapley interaction index over agent coalitions (paper Eq. 22-23; plan Section 8).

v(S) = (1 / T_eval) * sum over evaluation days of the Spearman IC of the scheduler-reweighted score
that uses only the agents in S (weights renormalized within S); v(empty) = 0.
I(i, j) = sum over S in [n] minus {i, j} of |S|! (n - |S| - 2)! / (n - 1)! * Delta_ij v(S).
"""

from __future__ import annotations

from math import factorial

import numpy as np

from detmaso.losses.stats import daily_spearman


def coalition_values(alphas: np.ndarray, logits: np.ndarray, labels: np.ndarray, label_ok: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """v indexed by bitmask (0 .. 2^n - 1), plus the number of degenerate days per coalition.

    alphas: [T, n, N]; logits: [T, n] scheduler logits. Renormalizing w within S equals a softmax over
    S's logits, which cannot underflow to 0/0. A day on which the coalition's score is constant (for
    example, only dead agents; DEVIATIONS 4) has IC 0 and still counts in T_eval.
    """
    T, n = alphas.shape[0], alphas.shape[1]
    v = np.zeros(2**n)
    degenerate = np.zeros(2**n, dtype=int)
    for mask in range(1, 2**n):
        members = [i for i in range(n) if mask >> i & 1]
        z = logits[:, members]
        ws = np.exp(z - z.max(axis=1, keepdims=True))
        ws /= ws.sum(axis=1, keepdims=True)
        score = np.einsum("ts,tsn->tn", ws, alphas[:, members, :])
        ic = daily_spearman(score, labels, label_ok)
        bad = ~np.isfinite(ic)
        degenerate[mask] = int(bad.sum())
        ic[bad] = 0.0
        v[mask] = float(ic.sum() / T)
    return v, degenerate


def shapley_weight(s: int, n: int) -> float:
    return factorial(s) * factorial(n - s - 2) / factorial(n - 1)


def interaction_index(v: np.ndarray, n: int) -> np.ndarray:
    """Grabisch-Roubens pairwise Shapley interaction index; NaN on the diagonal."""
    out = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            others = [k for k in range(n) if k not in (i, j)]
            total = 0.0
            for sub in range(2 ** len(others)):
                S = sum(1 << others[b] for b in range(len(others)) if sub >> b & 1)
                s = bin(S).count("1")
                delta = v[S | 1 << i | 1 << j] - v[S | 1 << i] - v[S | 1 << j] + v[S]
                total += shapley_weight(s, n) * delta
            out[i, j] = total
    return out
