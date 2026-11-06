"""Differentiable soft rank (Blondel et al. 2020, "Fast Differentiable Sorting and Ranking").

L2-regularized soft ranks, ascending (the largest input gets the largest rank, ranks lie in
[1, n]), matching torchsort's ``soft_rank(x, "l2", eps)`` convention:

    soft_rank(x) = argmin_{mu in Permutahedron(n, n-1, ..., 1)} 0.5 * || mu - x / eps ||^2

Computed exactly with the pool-adjacent-violators (PAV) algorithm (Blondel et al., Prop. 3).
The forward pass runs on CPU in the input dtype (use float64; see plan A20); the backward pass
is the exact Jacobian-vector product: within each pooled block the gradient is centered.

Regimes for cross-sectionally z-scored inputs (plan D4 / A21):
  * eps tiny  -> every gap between sorted inputs exceeds eps -> exact hard ranks (zero gradient)
  * eps large -> everything pools into one block -> soft rank is an affine copy of the input
"""

from __future__ import annotations

import numpy as np
import torch


def pav_nonincreasing(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """L2 isotonic regression with a non-increasing constraint, via pool-adjacent-violators.

    Solves argmin_v 0.5 * ||v - y||^2 subject to v[0] >= v[1] >= ... >= v[n-1].
    Returns (solution, block_id) where block_id[i] identifies the pooled block of element i.
    """
    sums: list[float] = []
    counts: list[int] = []
    for value in y.tolist():
        sums.append(value)
        counts.append(1)
        # Merge while the previous block's mean is below the last block's mean (a violation).
        while len(sums) > 1 and sums[-2] * counts[-1] < sums[-1] * counts[-2]:
            s = sums.pop()
            c = counts.pop()
            sums[-1] += s
            counts[-1] += c
    n = y.shape[0]
    solution = np.empty(n, dtype=np.float64)
    block_id = np.empty(n, dtype=np.int64)
    start = 0
    for b, (s, c) in enumerate(zip(sums, counts)):
        solution[start : start + c] = s / c
        block_id[start : start + c] = b
        start += c
    return solution, block_id


class _SoftRankL2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, eps: float) -> torch.Tensor:  # type: ignore[override]
        if x.dim() != 2:
            raise ValueError("soft_rank expects a 2-D tensor of shape (batch, n)")
        n = x.shape[1]
        theta = x / eps
        s, perm = torch.sort(theta, dim=1, descending=True)
        w = torch.arange(n, 0, -1, dtype=x.dtype, device=x.device)  # (n, n-1, ..., 1)
        y = (s - w).detach().cpu().numpy().astype(np.float64)
        solutions, blocks = zip(*(pav_nonincreasing(row) for row in y))
        v = torch.from_numpy(np.stack(solutions)).to(dtype=x.dtype, device=x.device)
        block = torch.from_numpy(np.stack(blocks)).to(device=x.device)
        r_sorted = s - v
        ranks = torch.empty_like(r_sorted).scatter_(1, perm, r_sorted)
        ctx.save_for_backward(perm, block)
        ctx.eps = eps
        return ranks

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        perm, block = ctx.saved_tensors
        g_sorted = grad_out.gather(1, perm)
        n_blocks = int(block.max().item()) + 1
        shape = (g_sorted.shape[0], n_blocks)
        sums = torch.zeros(shape, dtype=g_sorted.dtype, device=g_sorted.device)
        sums.scatter_add_(1, block, g_sorted)
        counts = torch.zeros(shape, dtype=g_sorted.dtype, device=g_sorted.device)
        counts.scatter_add_(1, block, torch.ones_like(g_sorted))
        block_means = (sums / counts.clamp_min(1.0)).gather(1, block)
        grad_sorted = g_sorted - block_means
        grad_theta = torch.empty_like(grad_sorted).scatter_(1, perm, grad_sorted)
        return grad_theta / ctx.eps, None


def soft_rank(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Ascending L2 soft ranks of each row of ``x`` (shape (batch, n)), ranks in [1, n]."""
    if eps <= 0:
        raise ValueError("eps must be positive")
    return _SoftRankL2.apply(x, float(eps))


def affine_threshold(z: torch.Tensor) -> torch.Tensor:
    """Smallest eps at which the soft rank of each row is exactly affine (one pooled block).

    For a row sorted descending, PAV pools everything iff every prefix mean of
    (theta/eps - w) is <= the overall mean, i.e. eps >= max_k 2 * (mean_topk - mean) / (n - k).
    """
    s, _ = torch.sort(z, dim=1, descending=True)
    n = s.shape[1]
    k = torch.arange(1, n, dtype=s.dtype, device=s.device)
    prefix_mean = torch.cumsum(s, dim=1)[:, :-1] / k
    overall = s.mean(dim=1, keepdim=True)
    return (2.0 * (prefix_mean - overall) / (n - k)).amax(dim=1)
