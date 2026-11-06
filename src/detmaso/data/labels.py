"""Prediction target: the tau-day forward return (paper l.204, tau = 20)."""

from __future__ import annotations

import pandas as pd


def forward_return(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """r_t = Close(t + horizon) / Close(t) - 1 on adjusted closes (row t = entry day).

    NaN if either end-point close is missing (A8 rule 4). Computed from each stock's full price
    history, so a stock that leaves the index within the horizon keeps its label (A3).
    """
    return close.shift(-horizon) / close - 1.0


def next_day_return(close: pd.DataFrame) -> pd.DataFrame:
    """Close(t + 1) / Close(t) - 1, used by the daily-rebalanced backtest; NaN if either is missing."""
    return close.shift(-1) / close - 1.0
