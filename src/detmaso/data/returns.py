"""Suspension-aware daily returns and the rolling-window validity rule (plan A8)."""

from __future__ import annotations

import math

import pandas as pd


def min_obs(window: int, coverage: float) -> int:
    """A rolling statistic is valid only if at least ``coverage`` of the window has data."""
    return max(1, math.ceil(coverage * window))


def prev_traded_close(close: pd.DataFrame) -> pd.DataFrame:
    """For each day, the last traded close strictly before that day."""
    return close.ffill().shift(1)


def returns_own_days(close: pd.DataFrame) -> pd.DataFrame:
    """Close over the previous *traded* close, minus 1 (A8 rule 1).

    The day trading resumes after a suspension carries the whole move; suspended days are NaN
    (never a fake 0% return, which pandas' default pct_change forward-fill would create).
    """
    ret = close / prev_traded_close(close) - 1.0
    return ret.where(close.notna())


def returns_one_day(close: pd.DataFrame) -> pd.DataFrame:
    """Close-to-close return only when the stock traded on both d-1 and d (A12)."""
    return close / close.shift(1) - 1.0
