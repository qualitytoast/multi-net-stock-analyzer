"""The paper's 34 stock features (Table 2), per plan Section 3.2.

Every feature is first computed with windows that END on day d (inclusive), then shifted by
exactly one row at the end, so row t only uses data through t - 1 (plan A5).
Inputs are wide (date x code) frames of Qlib's adjusted fields; NaN means "did not trade".
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import rankdata

from detmaso.data.returns import min_obs, prev_traded_close, returns_own_days

FEATURE_NAMES: tuple[str, ...] = (
    "mom_1", "mom_5", "mom_10", "mom_20", "mom_60", "mom_120",
    "vol_5", "vol_10", "vol_20", "vol_60",
    "rturn_5", "rturn_10", "rturn_20", "rturn_60",
    "pvcorr_5", "pvcorr_10", "pvcorr_20",
    "upfrac_5", "upfrac_10", "upfrac_20",
    "madev_5", "madev_20", "madev_60",
    "macd",
    "overnight", "intraday", "closepos", "hl_1", "hl_5",
    "skew60", "ac1_60", "momsharpe", "range20", "logvol20",
)  # fmt: skip
assert len(FEATURE_NAMES) == 34


def rolling_spearman(a: pd.DataFrame, b: pd.DataFrame, window: int, min_count: int, chunk: int = 256) -> pd.DataFrame:
    """Column-wise rolling Spearman correlation of a and b over windows ending at each row.

    Uses only days where both series exist; NaN if fewer than ``min_count`` such days or if
    either series is constant in the window.
    """
    A = a.to_numpy(dtype=np.float64)
    B = b.to_numpy(dtype=np.float64)
    T, N = A.shape
    out = np.full((T, N), np.nan)
    if T < window:
        return pd.DataFrame(out, index=a.index, columns=a.columns)
    for c0 in range(0, N, chunk):
        c1 = min(N, c0 + chunk)
        aw = sliding_window_view(A[:, c0:c1], window, axis=0).copy()  # (T-w+1, n, w)
        bw = sliding_window_view(B[:, c0:c1], window, axis=0).copy()
        both = ~np.isnan(aw) & ~np.isnan(bw)
        aw[~both] = np.nan
        bw[~both] = np.nan
        ra = rankdata(aw, axis=-1, nan_policy="omit")
        rb = rankdata(bw, axis=-1, nan_policy="omit")
        count = both.sum(axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows -> NaN, handled below
            ma = np.nanmean(np.where(both, ra, np.nan), axis=-1, keepdims=True)
            mb = np.nanmean(np.where(both, rb, np.nan), axis=-1, keepdims=True)
            da = np.where(both, ra - ma, 0.0)
            db = np.where(both, rb - mb, 0.0)
            cov = (da * db).sum(axis=-1)
            var_a = (da * da).sum(axis=-1)
            var_b = (db * db).sum(axis=-1)
            corr = cov / np.sqrt(var_a * var_b)
        corr[(count < min_count) | (var_a <= 0) | (var_b <= 0)] = np.nan
        out[window - 1 :, c0:c1] = corr
    return pd.DataFrame(out, index=a.index, columns=a.columns)


def compute_features(fields: dict[str, pd.DataFrame], coverage: float = 0.8) -> dict[str, pd.DataFrame]:
    """All 34 features, already shifted so that row t uses data through t - 1."""
    C, O, H, L = fields["close"], fields["open"], fields["high"], fields["low"]
    V = fields["volume"]  # split-adjusted volume (ratio features)
    v_raw = fields["volume"] * fields["factor"]  # raw volume in lots (level feature, A4)

    def mp(w: int) -> int:
        return min_obs(w, coverage)

    ret = returns_own_days(C)
    c_last = C.ffill()  # last traded close at or before each day (point-to-point features)
    prev_close = prev_traded_close(C)

    f: dict[str, pd.DataFrame] = {}
    # 1. Return momentum (compounded, A3.2 rows 1-6)
    for w in (1, 5, 10, 20, 60, 120):
        f[f"mom_{w}"] = c_last / c_last.shift(w) - 1.0
    # 2. Realized volatility
    for w in (5, 10, 20, 60):
        f[f"vol_{w}"] = ret.rolling(w, min_periods=mp(w)).std()
    # 3. Volume momentum ("relative turnover"): latest volume over its w-day mean
    for w in (5, 10, 20, 60):
        f[f"rturn_{w}"] = V / V.rolling(w, min_periods=mp(w)).mean()
    # 4. Price-volume Spearman correlation
    for w in (5, 10, 20):
        f[f"pvcorr_{w}"] = rolling_spearman(C, V, w, mp(w))
    # 5. RSI proxy: fraction of up days among traded days
    up = (ret > 0).astype("float64").where(ret.notna())
    for w in (5, 10, 20):
        f[f"upfrac_{w}"] = up.rolling(w, min_periods=mp(w)).mean()
    # 6. Moving-average deviation
    for w in (5, 20, 60):
        f[f"madev_{w}"] = c_last / C.rolling(w, min_periods=mp(w)).mean() - 1.0
    # 7. MACD-type (stock-level mu_20 - mu_60)
    f["macd"] = f["mom_20"] - f["mom_60"]
    # 8. Intraday structure
    f["overnight"] = O / prev_close - 1.0
    f["intraday"] = C / O - 1.0
    rng = H - L
    f["closepos"] = ((C - L) / rng).where(rng > 0, 0.5).where(C.notna())
    hl = rng / C
    f["hl_1"] = hl
    f["hl_5"] = hl.rolling(5, min_periods=mp(5)).mean()
    # 9. Statistical moments
    f["skew60"] = ret.rolling(60, min_periods=mp(60)).skew()
    f["ac1_60"] = ret.rolling(60, min_periods=mp(60)).corr(ret.shift(1))
    f["momsharpe"] = f["mom_60"] / (f["vol_60"] + 1e-12)
    f["range20"] = (H.rolling(20, min_periods=mp(20)).max() - L.rolling(20, min_periods=mp(20)).min()) / c_last
    f["logvol20"] = np.log(v_raw.rolling(20, min_periods=mp(20)).mean() + 1.0)

    missing = set(FEATURE_NAMES) - set(f)
    assert not missing, missing
    # Exactly one shift: row t sees data through t - 1 (plan A5).
    return {name: f[name].replace([np.inf, -np.inf], np.nan).shift(1) for name in FEATURE_NAMES}
