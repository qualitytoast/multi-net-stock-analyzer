"""The Det-MASO training loss for one date (paper Eq. 20; plan Sections 6.2-6.3, A20-A27)."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

from detmaso.config import LossConfig
from detmaso.losses.softrank import affine_threshold, soft_rank
from detmaso.losses.stats import hard_rank, zscore_rows


def rank_eps(kappa_soft: float, n_t: int) -> float:
    """Soft-rank regularization strength: kappa_soft scaled by the cross-section size (l.601, A20)."""
    return kappa_soft * n_t / 100.0


@dataclass
class SharpeState:
    """Detached history of the long-short portfolio returns p_s for the Sharpe window (A24).

    window=None: expanding window over every date processed so far in the epoch (T_train = number
    of training dates). Otherwise a fixed rolling window of that many dates (including today).
    """

    window: int | None = None
    n: int = 0
    s1: float = 0.0
    s2: float = 0.0
    recent: deque = field(default_factory=deque)

    def reset(self) -> None:
        self.n, self.s1, self.s2 = 0, 0.0, 0.0
        self.recent.clear()

    def push(self, p: float) -> None:
        if self.window is None:
            self.n += 1
            self.s1 += p
            self.s2 += p * p
        else:
            self.recent.append(p)
            while len(self.recent) > self.window - 1:
                self.recent.popleft()

    def sums(self) -> tuple[int, float, float]:
        if self.window is None:
            return self.n, self.s1, self.s2
        arr = np.asarray(self.recent, dtype=np.float64)
        return len(arr), float(arr.sum()), float((arr * arr).sum())


def neg_logdet(sigma: torch.Tensor, shrink: float) -> tuple[torch.Tensor, int]:
    """-log det((1 - shrink) * Sigma + shrink * I) via Cholesky (A23). Returns (value, retries)."""
    n = sigma.shape[-1]
    eye = torch.eye(n, dtype=sigma.dtype, device=sigma.device)
    sym = 0.5 * (sigma + sigma.transpose(-1, -2))
    retries = 0
    delta = shrink
    while True:
        mat = (1.0 - delta) * sym + delta * eye
        chol, info = torch.linalg.cholesky_ex(mat)
        if int(info) == 0:
            return -2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1), retries
        retries += 1
        delta = min(1.0, delta * 10.0)
        if retries > 6:
            raise FloatingPointError("Cholesky failed even with heavy shrinkage")


@dataclass
class LossTerms:
    loss: torch.Tensor
    ic_term: torch.Tensor  # sum_i w_i m_i IC_i
    sr: torch.Tensor
    neg_logdet: torch.Tensor
    entropy: torch.Tensor
    p_t: float
    ic: np.ndarray  # per-agent soft-rank IC
    mean_abs_corr: float  # unshrunk Sigma, off-diagonal
    logdet_raw: float  # log det of the unshrunk Sigma (reporting only)
    lambda_min: float
    affine_share: float  # share of agents whose soft rank was exactly affine today
    cholesky_retries: int
    n_dead: int  # agents with a constant (zeroed) score today


class DetMASOLoss:
    def __init__(self, cfg: LossConfig, n_agents: int):
        self.cfg = cfg
        self.n_agents = n_agents

    def __call__(
        self,
        alphas: torch.Tensor,  # [n, N] (one date)
        alpha_hat: torch.Tensor,  # [N]
        logits: torch.Tensor,  # [n]
        w: torch.Tensor,  # [n]
        y: torch.Tensor,  # [N] 20-day forward returns
        label_ok: torch.Tensor,  # [N] bool
        regime: int | None,  # agent index of today's regime; None = no regime mask (single network)
        sharpe: SharpeState,
    ) -> LossTerms:
        c = self.cfg
        idx = torch.nonzero(label_ok, as_tuple=False).squeeze(-1)
        a = alphas[:, idx].double()
        n_t = a.shape[1]
        eps = rank_eps(c.kappa_soft, n_t)
        ranks = soft_rank(a, eps)
        A = zscore_rows(ranks)  # standardized soft ranks, reused for IC and Sigma (l.229)
        y_l = y[idx].double()
        Y = zscore_rows(torch.as_tensor(hard_rank(y_l.detach().cpu().numpy()), dtype=torch.float64, device=a.device).unsqueeze(0))[0]
        ic = (A * Y).mean(dim=1)  # [n]
        sigma = A @ A.T / n_t  # [n, n], unit diagonal
        nld, retries = neg_logdet(sigma, c.shrink)

        # Sharpe of the long-short portfolio with weights alpha_hat (Eq. 12); past p_s detached.
        p_t = (alpha_hat[idx].double() * y_l).sum()
        n_prev, s1, s2 = sharpe.sums()
        n_now = n_prev + 1
        if n_now >= c.sharpe_min_points:
            mean = (s1 + p_t) / n_now
            var = ((s2 + p_t * p_t) - n_now * mean * mean) / (n_now - 1)
            sr = mean / (torch.sqrt(torch.clamp(var, min=0.0)) + c.sharpe_eps)
        else:
            sr = torch.zeros((), dtype=torch.float64, device=a.device)

        log_w = torch.log_softmax(logits.double(), dim=-1)
        entropy = -(log_w.exp() * log_w).sum()
        if regime is None:
            mask = torch.ones(self.n_agents, dtype=torch.float64, device=a.device)
        else:
            mask = torch.full((self.n_agents,), c.lambda_mask, dtype=torch.float64, device=a.device)
            mask[regime] = 1.0
        ic_term = (w.double() * mask * ic).sum()
        loss = -c.c_ic * ic_term - c.c_sr * sr + c.c_div * nld - c.c_ent * entropy

        with torch.no_grad():
            live = A.abs().sum(dim=1) > 0  # agents with a non-constant score today
            sub = sigma[live][:, live]
            k = int(live.sum())
            off = sub[~torch.eye(k, dtype=torch.bool, device=sigma.device)] if k > 1 else sub.new_zeros(1)
            sign, logabs = torch.linalg.slogdet(sub)
            eig = torch.linalg.eigvalsh(0.5 * (sub + sub.T))
            affine = (affine_threshold(a) <= eps).double().mean()
        return LossTerms(
            loss=loss,
            ic_term=ic_term,
            sr=sr,
            neg_logdet=nld,
            entropy=entropy,
            p_t=float(p_t.detach()),
            ic=ic.detach().cpu().numpy(),
            mean_abs_corr=float(off.abs().mean()),
            logdet_raw=float(logabs) if float(sign) > 0 else -math.inf,
            lambda_min=float(eig.min()) if k > 0 else float("nan"),
            affine_share=float(affine),
            cholesky_retries=retries,
            n_dead=self.n_agents - k,
        )
