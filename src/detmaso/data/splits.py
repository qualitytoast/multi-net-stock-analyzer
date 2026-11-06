"""Train / validation / test date splits with the 20-day purge (paper l.725; plan A6, A7)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Splits:
    train: pd.DatetimeIndex  # all training dates (used for the scaler / K-means)
    train_fit: pd.DatetimeIndex  # training dates whose labels end inside the training period
    valid: pd.DatetimeIndex
    valid_fit: pd.DatetimeIndex  # validation dates whose labels end inside the validation period
    test: pd.DatetimeIndex


def _between(dates: pd.DatetimeIndex, start: str, end: str) -> pd.DatetimeIndex:
    return dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]


def make_splits(dates: pd.DatetimeIndex, train, valid, test, purge_days: int) -> Splits:
    tr = _between(dates, *train)
    va = _between(dates, *valid)
    te = _between(dates, *test)
    # A label at t uses Close(t + 20); dropping the last `purge_days` dates keeps every label
    # inside its own split (t + 20 <= last trading day of the split).
    return Splits(
        train=tr,
        train_fit=tr[:-purge_days] if purge_days else tr,
        valid=va,
        valid_fit=va[:-purge_days] if purge_days else va,
        test=te,
    )
