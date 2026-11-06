"""The 12-dim multi-scale market state s_t (paper Table 1, Sec 3.2.1/3.2.3; plan A12)."""

from __future__ import annotations

import pandas as pd

from detmaso.data.returns import min_obs, returns_one_day


def state_names(windows: tuple[int, ...]) -> list[str]:
    """Grouped by window, as the scheduler's per-window encoders expect: [sigma, mu, delta] x w."""
    return [f"{stat}{w}" for w in windows for stat in ("sigma", "mu", "delta")]


def daily_market_series(close: pd.DataFrame, member: pd.DataFrame) -> pd.DataFrame:
    """Per day d: equal-weighted member return and cross-sectional dispersion of member returns.

    Only members that traded on both d-1 and d contribute (no fake 0% days around suspensions).
    """
    r = returns_one_day(close).where(member)
    return pd.DataFrame({"r_mkt": r.mean(axis=1), "disp": r.std(axis=1, ddof=1)})


def market_state(close: pd.DataFrame, member: pd.DataFrame, windows: tuple[int, ...], coverage: float = 0.8) -> pd.DataFrame:
    """s_t from data in [t - w, t - 1]: rolling stats over windows ending at d, then one shift."""
    daily = daily_market_series(close, member)
    cols = {}
    for w in windows:
        mp = min_obs(w, coverage)
        cols[f"sigma{w}"] = daily["r_mkt"].rolling(w, min_periods=mp).std()  # realized volatility
        cols[f"mu{w}"] = daily["r_mkt"].rolling(w, min_periods=mp).sum()  # time-series momentum
        cols[f"delta{w}"] = daily["disp"].rolling(w, min_periods=mp).mean()  # dispersion
    return pd.DataFrame(cols)[state_names(windows)].shift(1)
