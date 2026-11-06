"""Plan tests 13-21: look-ahead, labels, suspensions, splits, regimes, membership, processing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from detmaso.data.features import FEATURE_NAMES, compute_features, rolling_spearman
from detmaso.data.labels import forward_return
from detmaso.data.market_state import market_state
from detmaso.data.panel import cross_sectional_process
from detmaso.data.returns import returns_one_day, returns_own_days
from detmaso.data.splits import make_splits
from tests.conftest import synthetic_fields

D_POS = 260  # row index of "day t" used by the look-ahead tests (well after warm-up)


def _shock(fields: dict[str, pd.DataFrame], rows: slice, rng_seed: int = 9) -> dict[str, pd.DataFrame]:
    """Multiply every field on the given rows by strong random factors (prices and volume)."""
    rng = np.random.default_rng(rng_seed)
    out = {k: v.copy() for k, v in fields.items()}
    n_rows = len(out["close"].iloc[rows])
    scale_px = rng.uniform(1.5, 3.0, size=(n_rows, out["close"].shape[1]))
    for k in ("open", "high", "low", "close"):
        out[k].iloc[rows] = out[k].iloc[rows].to_numpy() * scale_px
    out["high"].iloc[rows] = out["high"].iloc[rows].to_numpy() * 1.05  # change the intraday range too
    out["open"].iloc[rows] = out["open"].iloc[rows].to_numpy() * 0.93  # and the open-to-close move
    out["volume"].iloc[rows] = out["volume"].iloc[rows].to_numpy() * rng.uniform(3, 6, size=scale_px.shape)
    return out


# 13 ------------------------------------------------------------------------------------------
def test_features_do_not_see_day_t_or_later():
    f = synthetic_fields()
    base = compute_features(f)
    shocked = compute_features(_shock(f, slice(D_POS, None)))
    for name in FEATURE_NAMES:
        pd.testing.assert_frame_equal(base[name].iloc[: D_POS + 1], shocked[name].iloc[: D_POS + 1], obj=name)


def test_features_do_see_day_t_minus_1():
    f = synthetic_fields()
    base = compute_features(f)
    shocked = compute_features(_shock(f, slice(D_POS - 1, D_POS)))
    unchanged = [n for n in FEATURE_NAMES if np.allclose(base[n].iloc[D_POS], shocked[n].iloc[D_POS], equal_nan=True)]
    assert not unchanged, f"features ignoring day t-1 (double shift?): {unchanged}"


# 14 ------------------------------------------------------------------------------------------
def test_market_state_uses_exactly_days_before_t():
    f = synthetic_fields()
    member = pd.DataFrame(True, index=f["close"].index, columns=f["close"].columns)
    base = market_state(f["close"], member, (5, 20, 60, 120))
    later = market_state(_shock(f, slice(D_POS, None))["close"], member, (5, 20, 60, 120))
    prev = market_state(_shock(f, slice(D_POS - 1, D_POS))["close"], member, (5, 20, 60, 120))
    pd.testing.assert_frame_equal(base.iloc[: D_POS + 1], later.iloc[: D_POS + 1])
    assert not np.allclose(base.iloc[D_POS], prev.iloc[D_POS])


# 15 ------------------------------------------------------------------------------------------
def test_scores_and_picks_ignore_labels_and_future_prices():
    """Wiping all prices after day t (hence every label) leaves day-t inputs and masks unchanged."""
    import torch

    from detmaso.eval.backtest import top_decile_backtest
    from detmaso.model.detmaso import DetMASO

    f = synthetic_fields()
    wiped = {k: v.copy() for k, v in f.items()}
    for k in wiped:
        wiped[k].iloc[D_POS + 1 :] = np.nan
    member = pd.DataFrame(True, index=f["close"].index, columns=f["close"].columns)

    def day_inputs(fields):
        feats = compute_features(fields)
        traded = fields["close"].notna()
        universe = member & traded.shift(1, fill_value=False)
        tradable = universe & traded
        X = np.stack([cross_sectional_process(feats[n], universe, 5.0).iloc[D_POS].fillna(0).to_numpy() for n in FEATURE_NAMES], -1)
        s = market_state(fields["close"], member, (5, 20, 60, 120)).iloc[D_POS].to_numpy()
        return X, universe.iloc[D_POS].to_numpy(), tradable.iloc[D_POS].to_numpy(), s

    X1, U1, T1, s1 = day_inputs(f)
    X2, U2, T2, s2 = day_inputs(wiped)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(U1, U2)
    np.testing.assert_array_equal(T1, T2)
    np.testing.assert_array_equal(s1, s2)
    assert np.isnan(forward_return(wiped["close"], 20).iloc[D_POS]).all()  # labels really are gone

    torch.manual_seed(0)
    model = DetMASO(5, 34, (8, 4), activation="gelu").double()
    to = lambda a: torch.as_tensor(a, dtype=torch.float64).unsqueeze(0)  # noqa: E731
    out1 = model(to(X1), torch.as_tensor(U1).unsqueeze(0), to(s1))
    out2 = model(to(X2), torch.as_tensor(U2).unsqueeze(0), to(s2))
    assert torch.equal(out1.alpha_hat, out2.alpha_hat)
    r_next = np.random.default_rng(0).normal(size=(1, len(U1)))
    _, _, p1 = top_decile_backtest(out1.alpha_hat.detach().numpy(), T1[None], r_next, 0.25)
    _, _, p2 = top_decile_backtest(out2.alpha_hat.detach().numpy(), T2[None], r_next, 0.25)
    np.testing.assert_array_equal(p1[0], p2[0])


# 16 ------------------------------------------------------------------------------------------
def test_label_alignment_on_hand_built_series():
    dates = pd.bdate_range("2021-01-01", periods=30)
    close = pd.DataFrame({"A": np.arange(1, 31, dtype=float)}, index=dates)
    label = forward_return(close, 20)
    assert label.iloc[0, 0] == pytest.approx(21 / 1 - 1)
    assert label.iloc[9, 0] == pytest.approx(30 / 10 - 1)
    assert label.iloc[10:].isna().all().all()


# 17 ------------------------------------------------------------------------------------------
def test_suspension_handling():
    dates = pd.bdate_range("2021-01-01", periods=8)
    close = pd.DataFrame({"A": [10, 11, np.nan, np.nan, 12.1, 12.1, 13.31, 13.31], "B": [5.0] * 8}, index=dates)
    own = returns_own_days(close)["A"]
    assert own.iloc[4] == pytest.approx(0.1)  # resumption day carries the whole move
    assert own.iloc[2:4].isna().all()  # no fake 0% days
    one_day = returns_one_day(close)["A"]
    assert np.isnan(one_day.iloc[4])  # market state skips stocks that did not trade on both days
    rolling = own.rolling(5, min_periods=4).std()
    assert np.isnan(rolling.iloc[5])  # only 3 valid returns in the window ending at row 5


def test_rolling_spearman_matches_direct_computation():
    from scipy.stats import spearmanr

    f = synthetic_fields()
    r = rolling_spearman(f["close"], f["volume"], 10, 8)
    window = slice(D_POS - 9, D_POS + 1)
    for col in f["close"].columns[:3]:
        expected = spearmanr(f["close"][col].iloc[window], f["volume"][col].iloc[window]).statistic
        assert r[col].iloc[D_POS] == pytest.approx(expected)


# 18 ------------------------------------------------------------------------------------------
def test_splits_respect_the_20_day_horizon():
    dates = pd.bdate_range("2020-01-01", "2022-12-31")
    s = make_splits(dates, ("2020-01-01", "2020-12-31"), ("2021-01-01", "2021-06-30"), ("2021-07-01", "2022-12-31"), 20)
    pos = {d: i for i, d in enumerate(dates)}
    assert dates[pos[s.train_fit[-1]] + 20] <= s.train[-1]
    assert dates[pos[s.train_fit[-1]] + 21] > s.train[-1]
    assert dates[pos[s.valid_fit[-1]] + 20] <= s.valid[-1]
    assert not set(s.train) & set(s.valid) and not set(s.valid) & set(s.test)


@pytest.mark.data
def test_real_splits_and_frozen_regimes(real):
    cfg, panel, regimes, splits = real
    assert pd.Timestamp(regimes.fit_end) <= pd.Timestamp(cfg.splits.train[1])
    cal = panel.dates
    pos = cal.get_indexer(splits.train_fit[-1:])[0]
    assert cal[pos + 20] <= splits.train[-1]
    assert not set(splits.train) & set(splits.test)


# 19 ------------------------------------------------------------------------------------------
def test_regimes_fit_on_training_dates_only():
    from detmaso.data.regimes import fit_regimes

    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=400)
    state = pd.DataFrame(rng.normal(size=(400, 12)), index=dates, columns=[f"{s}{w}" for w in (5, 20, 60, 120) for s in ("sigma", "mu", "delta")])
    train = dates[:300]
    model = fit_regimes(state, train, n_agents=5, k_range=(2, 6), n_init=5, seed=0)
    assert pd.Timestamp(model.fit_end) == train[-1]
    shocked = state.copy()
    shocked.iloc[300:] *= 100  # wildly different "future" must not change the fit
    model2 = fit_regimes(shocked, train, n_agents=5, k_range=(2, 6), n_init=5, seed=0)
    np.testing.assert_array_equal(model.centroids, model2.centroids)


# 20 ------------------------------------------------------------------------------------------
@pytest.mark.data
def test_real_membership(real):
    cfg, panel, regimes, splits = real
    from detmaso.data.qlib_io import cache_dir

    member = pd.read_parquet(cache_dir(cfg) / "member.parquet")
    for d in ["2012-01-04", "2022-12-30", "2024-03-01", "2026-02-27"]:
        assert member.loc[d].sum() == 300
    # Every universe stock on a date is an index member that day.
    for t in [0, 1000, 2000, len(panel.dates) - 1]:
        codes = [panel.code_list[i] for i in panel.code_idx[t][panel.universe[t]].tolist()]
        assert member.loc[panel.dates[t], codes].all()
    # A stock leaving the index within 20 days keeps its label.
    leaving = member.astype(int).diff().eq(-1)
    d, code = next((d, c) for d in leaving.index[leaving.any(axis=1)] for c in leaving.columns[leaving.loc[d]] if d > panel.dates[200])
    t = panel.dates.get_indexer([d])[0] - 5
    row = panel.code_idx[t].tolist()
    k = row.index(panel.code_list.index(code))
    assert bool(panel.label_ok[t, k])


# 21 ------------------------------------------------------------------------------------------
def test_cross_sectional_processing():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2021-01-01", periods=5)
    F = pd.DataFrame(rng.normal(size=(5, 30)), index=dates)
    F.iloc[0, 3] = np.nan
    F.iloc[1, 4] = 50.0  # outlier
    universe = pd.DataFrame(True, index=dates, columns=F.columns)
    universe.iloc[:, -5:] = False
    unclipped = cross_sectional_process(F, universe, clip=1e9)
    vals = unclipped.where(universe & F.notna())
    np.testing.assert_allclose(vals.mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(vals.std(axis=1, ddof=0), 1.0, atol=1e-12)
    clipped = cross_sectional_process(F, universe, clip=5.0)
    assert clipped.where(universe).abs().max().max() <= 5.0
    assert clipped.iloc[0, 3] == 0.0  # NaN inside the universe becomes exactly 0
    assert clipped.iloc[:, -5:].isna().all().all()  # non-members absent
