"""Evaluation metrics (paper Sec 4.1.4; plan A41, A42, Section 7)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ic_summary(ic: np.ndarray) -> dict[str, float]:
    """IC = mean of daily Spearman ICs; ICIR = mean / std (ddof 1), not annualized."""
    ic = ic[np.isfinite(ic)]
    std = float(ic.std(ddof=1)) if len(ic) > 1 else float("nan")
    return {"ic": float(ic.mean()), "ic_std": std, "icir": float(ic.mean() / std), "ic_days": int(len(ic))}


def portfolio_metrics(returns: np.ndarray, start: pd.Timestamp, end: pd.Timestamp, qlib_days: int = 238) -> dict[str, float]:
    """Headline (standard) and Qlib-style metrics for a daily return series.

    returns[k] is the return earned over day k of the holding period, which starts at the close
    of ``start`` and ends at the close of ``end``.

    Headline (A41): compounded annual growth over calendar time; Sharpe annualized with the
    trading days per year actually observed; max drawdown of the value curve with the starting
    value 1 counted as a peak; risk-free rate 0.
    Qlib form (qlib.contrib.evaluate.risk_analysis, mode='sum'): mean * N, mean / std * sqrt(N),
    min(cumsum - running max of cumsum), with N = 238.
    """
    r = np.asarray(returns, dtype=np.float64)
    years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    nav = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    peak = np.maximum.accumulate(nav)
    days_per_year = len(r) / years
    std = r.std(ddof=1)
    std = std if std > 0 else float("nan")
    cum = np.cumsum(r)
    return {
        "ar": float(nav[-1] ** (1.0 / years) - 1.0),
        "sharpe": float(r.mean() / std * np.sqrt(days_per_year)),
        "mdd": float((1.0 - nav / peak).max()),
        "total_return": float(nav[-1] - 1.0),
        "days_per_year": float(days_per_year),
        "qlib_ar": float(r.mean() * qlib_days),
        "qlib_ir": float(r.mean() / std * np.sqrt(qlib_days)),
        "qlib_mdd": float(-(cum - np.maximum.accumulate(cum)).min()),
    }
