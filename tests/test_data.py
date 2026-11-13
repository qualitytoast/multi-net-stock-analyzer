"""Plan tests 13-21: look-ahead, labels, suspensions, splits, regimes, membership, processing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from detmaso.data.features import FEATURE_NAMES, compute_features, rolling_spearman
from detmaso.data.labels import forward_return
from detmaso.data.market_state import market_state
from detmaso.data.panel import cross_sectional_process, make_masks, model_membership, seasoned_members, without_codes
from detmaso.data.returns import returns_one_day, returns_own_days
from detmaso.data.splits import make_splits
from tests.conftest import synthetic_config, synthetic_fields

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
    changed = ~np.isclose(base.iloc[D_POS].to_numpy(), prev.iloc[D_POS].to_numpy())
    assert changed.all(), base.columns[~changed].tolist()


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
        universe, tradable, label_ok, _ = make_masks(fields["close"], member, 20)
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
    assert make_masks(f["close"], member, 20)[2].iloc[D_POS].any()
    assert not make_masks(wiped["close"], member, 20)[2].iloc[D_POS].any()  # only label_ok changes

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


def test_window_coverage_rule_in_production_features():
    from detmaso.data.returns import min_obs

    assert min_obs(5, 0.8) == 4
    f = synthetic_fields(suspend=False)
    row = D_POS
    # Stock 0: 2 missing returns in the 5-day window ending at row-1 -> 3 valid -> vol_5 NaN.
    # Stock 2: 1 missing return -> 4 valid -> vol_5 finite.
    for k in f:
        f[k].iloc[row - 3, 0] = np.nan
        f[k].iloc[row - 5, 0] = np.nan
        f[k].iloc[row - 3, 2] = np.nan
    feats = compute_features(f)
    assert np.isnan(feats["vol_5"].iloc[row, 0])
    assert np.isfinite(feats["vol_5"].iloc[row, 2])


def test_market_series_uses_only_members_that_traded_both_days():
    from detmaso.data.market_state import daily_market_series

    dates = pd.bdate_range("2021-01-01", periods=6)
    close = pd.DataFrame({
        "A": [10.0, 10.0, 10.5, 10.5, 10.5, 10.5],
        "B": [20.0, np.nan, 40.0, 40.0, 40.0, 40.0],  # resumes on day 2 with a +100% move
        "C": [5.0, 5.0, 50.0, 50.0, 50.0, 50.0],      # +900% but not an index member
        "D": [8.0, 8.0, 8.4, 8.4, 8.4, 8.4],
    }, index=dates)  # fmt: skip
    member = pd.DataFrame(True, index=dates, columns=close.columns)
    member["C"] = False
    daily = daily_market_series(close, member)
    expected = np.mean([0.05, 0.05])  # A and D only: B did not trade on day 1, C is not a member
    assert daily["r_mkt"].iloc[2] == pytest.approx(expected)
    assert daily["disp"].iloc[2] == pytest.approx(0.0)


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


# Membership filters (docs/RULES.md, Rule CU) ------------------------------------------------------------------
def test_new_listing_filter_counts_traded_days_up_to_the_day_before():
    dates = pd.bdate_range("2021-01-01", periods=12)
    close = pd.DataFrame({
        "OLD": np.arange(1.0, 13.0),                                            # trades from the first loaded day
        "IPO": [np.nan, np.nan, 5.0, 9.0, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0, 10.1, 10.2],  # lists on row 2
        "GAP": [3.0, 3.1, np.nan, np.nan, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9],   # suspended on rows 2-3
    }, index=dates)  # fmt: skip
    member = pd.DataFrame(True, index=dates, columns=close.columns)
    member.loc[dates[:2], "IPO"] = False  # the list itself starts on the listing day
    out = seasoned_members(member, close, 3)
    # a stock enters on its 4th traded day: it then has 3 traded days up to the day before
    assert out["OLD"].tolist() == [False] * 3 + [True] * 9
    assert out["IPO"].tolist() == [False] * 5 + [True] * 7  # traded rows 2, 3, 4 -> member from row 5
    assert out["GAP"].tolist() == [False] * 5 + [True] * 7  # traded rows 0, 1, 4 -> member from row 5
    assert seasoned_members(member, close, 1)["IPO"].tolist() == [False] * 3 + [True] * 9  # never on its first traded day
    # it only ever removes member-days, and a stock that is not on the list stays out
    member.loc[dates[8:], "OLD"] = False
    out = seasoned_members(member, close, 3)
    assert not (out & ~member).to_numpy().any() and out["OLD"].tolist() == [False] * 3 + [True] * 5 + [False] * 4


def test_new_listing_filter_enters_on_the_121st_traded_day():
    f = synthetic_fields(suspend=False)
    close = f["close"].copy()
    close.iloc[:10, 4] = np.nan  # stock 4 lists on row 10 ...
    close.iloc[50:53, 4] = np.nan  # ... and is suspended for 3 days
    member = pd.DataFrame(True, index=close.index, columns=close.columns)
    out = seasoned_members(member, close, 120)
    assert out.iloc[:, 0].to_numpy().argmax() == 120  # rows 0..119 are its first 120 traded days
    first = out.iloc[:, 4].to_numpy().argmax()
    assert first == 10 + 120 + 3 and close.iloc[: first + 1, 4].notna().sum() == 121 and not out.iloc[:first, 4].any()


def test_new_listing_filter_does_not_look_ahead():
    """Membership on day t uses traded days up to t-1 only: nothing on t or later can change it."""
    f = synthetic_fields()
    close = f["close"].copy()
    close.iloc[:200, 5] = np.nan  # stock 5 lists on row 200: its 61st traded day is row 260
    member = pd.DataFrame(True, index=close.index, columns=close.columns)
    base = seasoned_members(member, close, 60)
    assert not base.iloc[259, 5] and base.iloc[260, 5]
    for t in (201, 230, 259, 260, 261, 300):
        wiped = close.copy()
        wiped.iloc[t:] = np.nan  # no price at all from day t on
        pd.testing.assert_frame_equal(base.iloc[: t + 1], seasoned_members(member, wiped, 60).iloc[: t + 1])
        shocked = _shock(f, slice(t, None))["close"]
        shocked.iloc[:200, 5] = np.nan
        pd.testing.assert_frame_equal(base, seasoned_members(member, shocked, 60))  # prices never matter, only trading
    # the day before does count: without a trade on row 259 the stock enters one day later
    late = close.copy()
    late.iloc[259, 5] = np.nan
    out = seasoned_members(member, late, 60)
    assert not out.iloc[260, 5] and out.iloc[261, 5]
    # and so the market state of day t still reads nothing from day t on
    state = market_state(close, base, (5, 20, 60, 120))
    wiped = close.copy()
    wiped.iloc[D_POS:] = np.nan
    later = market_state(wiped, seasoned_members(member, wiped, 60), (5, 20, 60, 120))
    pd.testing.assert_frame_equal(state.iloc[: D_POS + 1], later.iloc[: D_POS + 1])


def test_excluded_codes_are_never_members():
    dates = pd.bdate_range("2021-01-01", periods=4)
    member = pd.DataFrame(True, index=dates, columns=["SH000300", "SH600000", "SZ399300"])
    out = without_codes(member, ("SH000300", "SZ399300"))
    assert out.sum().to_dict() == {"SH000300": 0, "SH600000": 4, "SZ399300": 0} and member.to_numpy().all()  # the input is not changed
    with pytest.raises(ValueError, match="not in the loaded universe.*SH000905"):
        without_codes(member, ("SH000905",))


# A synthetic market prepared through the real pipeline (raw cache -> build -> data report) -------------------------
def _prepare(run_dir, fields, member, **overrides):
    """`detmaso prepare` without Qlib: the raw cache is written by hand, then build() reads it."""
    from detmaso.data.panel import build
    from detmaso.data.qlib_io import RawPanel, save_raw_panel

    overrides.setdefault("splits", {"train": ("2020-07-01", "2020-12-31")})  # January-June 2020 is warm-up
    overrides.setdefault("regimes", {"k_min": 5, "k_max": 5, "n_init": 2})
    cfg = synthetic_config(run_dir, variant="flagged" if "data" in overrides else "paper", **overrides)  # data flags are post-paper
    cal = fields["close"].index
    save_raw_panel(RawPanel(fields=fields, member=member, index_close=pd.Series(1.0, index=cal), calendar=cal), cfg)
    return cfg, build(cfg)


def _market(n_days: int = 520, n_stocks: int = 10):
    fields = synthetic_fields(n_days=n_days, n_stocks=n_stocks)
    member = pd.DataFrame(True, index=fields["close"].index, columns=fields["close"].columns)
    return fields, member


def _blank(fields, rows, col) -> None:
    for frame in fields.values():
        frame.iloc[rows, col] = np.nan


def test_flags_off_leave_the_membership_untouched(tmp_path):
    fields, member = _market()
    cfg, data = _prepare(tmp_path, fields, member)
    assert cfg.data.min_traded_days == 0 and cfg.data.exclude_codes == ()
    assert model_membership(data.raw, cfg) is data.raw.member and data.member is data.raw.member  # no filter code runs at all
    universe, tradable, label_ok, _ = make_masks(data.raw.fields["close"], data.raw.member, cfg.features.horizon)
    pd.testing.assert_frame_equal(data.masks["universe"], universe)
    pd.testing.assert_frame_equal(data.state, market_state(data.raw.fields["close"], data.raw.member, cfg.regimes.windows))


def test_new_listing_filter_keeps_ipo_days_out_of_the_universe_and_the_market_state(tmp_path):
    from detmaso.data.market_state import daily_market_series

    fields, member = _market()
    ipo, first = 9, 300  # stock S09 lists on row 300 (a model date) and jumps +400% on its second day
    _blank(fields, slice(0, first), ipo)
    member.iloc[:first, ipo] = False
    for name in ("open", "high", "low", "close"):
        fields[name].iloc[first + 1 :, ipo] *= 5.0
    cfg, data = _prepare(tmp_path / "on", fields, member, data={"min_traded_days": 5})
    off_cfg, off = _prepare(tmp_path / "off", fields, member)
    close, dates = data.raw.fields["close"], data.raw.calendar
    assert cfg.data_hash() != off_cfg.data_hash()  # a different prepared dataset
    # member, and in the universe, from its 6th traded day; with the flag off it is in the universe on its 2nd
    assert data.member.iloc[:, ipo].to_numpy().argmax() == first + 5 == data.masks["universe"].iloc[:, ipo].to_numpy().argmax()
    assert off.masks["universe"].iloc[:, ipo].to_numpy().argmax() == first + 1
    code = data.panel.code_list.index("S09")
    for t in range(first, first + 5):
        row = data.panel.dates.get_indexer([dates[t]])[0]
        assert code not in data.panel.code_idx[row][data.panel.universe[row]].tolist()
    # the +400% day never reaches the market series: it equals the series of the 9 other stocks
    others = daily_market_series(close.drop(columns="S09"), member.drop(columns="S09"))
    seasoned, polluted = daily_market_series(close, data.member), daily_market_series(close, off.member)
    assert seasoned.iloc[:5].isna().all().all()  # nobody has 5 traded days in the loaded data yet
    pd.testing.assert_frame_equal(seasoned.iloc[5 : first + 5], others.iloc[5 : first + 5])
    assert polluted["r_mkt"].iloc[first + 1] > others["r_mkt"].iloc[first + 1] + 0.3  # 400% / 10 stocks
    pd.testing.assert_frame_equal(data.state, market_state(close, data.member, cfg.regimes.windows))
    assert not np.allclose(data.state.iloc[first + 2], off.state.iloc[first + 2])


def test_excluded_codes_never_reach_the_universe_or_the_market_state(tmp_path):
    fields, member = _market()
    cfg, data = _prepare(tmp_path, fields, member, data={"exclude_codes": ("S03", "S07")})
    assert data.panel.code_list == list(member.columns)  # still in the raw cache and the code list
    used = set(data.panel.code_idx[data.panel.universe].tolist())
    assert used == {i for i, c in enumerate(member.columns) if c not in ("S03", "S07")}
    keep = [c for c in member.columns if c not in ("S03", "S07")]
    pd.testing.assert_frame_equal(data.state, market_state(data.raw.fields["close"][keep], member[keep], cfg.regimes.windows))


# Data QA report ---------------------------------------------------------------------------------------------
def _qa_market():
    """10 stocks, 520 days from 2020-01-01; model dates from 2020-07-01, test period from 2021-04-01 (row 326)."""
    fields, member = _market()  # S01 is suspended on rows 150-152 and S03 on row 200
    _blank(fields, slice(250, 290), 6)  # S06: a 40-day hole in the data while it is a member
    _blank(fields, slice(300, None), 5)  # S05: delisted, its last close is on row 299 ...
    member.iloc[310:, 5] = False  # ... and it leaves the list 10 days later
    member.iloc[:, 7] = False  # S07: never a member before the test period
    member.iloc[:200, 8] = False  # S08: joins on row 200
    return fields, member


def _report(run_dir, fields, member, **overrides) -> str:
    from detmaso.data.qa import write_data_report

    cfg, data = _prepare(run_dir, fields, member, **overrides)
    return write_data_report(data, cfg).read_text()


def _section(report: str, title: str) -> str:
    return report.split(f"\n{title}")[1].split("\n## ")[0]


def test_membership_stats_by_hand():
    from detmaso.data.qa import membership_stats, unpriced_spans

    dates = pd.bdate_range("2020-12-28", periods=8)  # 4 days in 2020, 4 in 2021
    member = pd.DataFrame({
        "A": [1, 1, 1, 1, 1, 1, 1, 1],  # priced throughout
        "B": [1, 1, 1, 1, 1, 1, 0, 0],  # leaves the list; its data ends on row 4
        "C": [0, 0, 1, 1, 1, 1, 1, 1],  # joins on row 2, never priced
        "D": [1, 1, 1, 1, 1, 1, 1, 1],  # unpriced on rows 1-2 and on the last row
        "E": [0, 0, 0, 0, 0, 0, 0, 0],  # never a member: its missing data counts nowhere
    }, index=dates).astype(bool)  # fmt: skip
    x = np.nan
    close = pd.DataFrame({
        "A": [1, 1, 1, 1, 1, 1, 1, 1], "B": [1, 1, 1, 1, 1, x, x, x], "C": [x] * 8, "D": [1, x, x, 1, 1, 1, 1, x], "E": [1, 1, x, x, x, x, x, x],
    }, index=dates, dtype=float)  # fmt: skip
    st = membership_stats(member, close, dates)
    y20, y21 = st["years"]
    assert (y20["year"], y20["days"], y20["min"], y20["median"], y20["max"]) == (2020, 4, 3, 3.5, 4)
    assert (y20["member_days"], y20["priced"]) == (14, 10)  # A 4, B 4, C 0 of 2, D 2 of 4
    assert (y21["year"], y21["days"], y21["min"], y21["median"], y21["max"]) == (2021, 4, 3, 3.5, 4)
    assert (y21["member_days"], y21["priced"]) == (14, 8)  # A 4, B 1 of 2, C 0 of 4, D 3 of 4
    # data ends early: B (last close 2021-01-01), C (no close), D (last close on the day before the last date)
    assert (st["stocks"], st["ended"], st["never_priced"]) == (4, 3, 1)
    assert (y20["ended_members"], y21["ended_members"]) == (3, 3) and (y20["last_close_in_year"], y21["last_close_in_year"]) == (0, 2)
    assert (st["member_days"], st["priced"], st["min"], st["median"], st["max"]) == (28, 18, 3, 3.5, 4)
    # spans, longest first: C rows 2-7, D rows 1-2, then B row 5 and D row 7 (ties: lower column first)
    assert st["spans"].tolist() == [[2, 2, 7], [3, 1, 2], [1, 5, 5], [3, 7, 7]]
    assert int((st["spans"][:, 2] - st["spans"][:, 1] + 1).sum()) == st["member_days"] - st["priced"]
    # a survivors-only dataset shows no early end and no span
    full = membership_stats(member[["A"]], close[["A"]], dates)
    assert full["ended"] == 0 and all(y["ended_members"] == y["last_close_in_year"] == 0 for y in full["years"])
    assert unpriced_spans(np.ones((3, 2), dtype=bool), np.ones((3, 2), dtype=bool)).shape == (0, 3)
    # only the given dates are read
    cut = membership_stats(member, close, dates[:4])
    assert cut["years"] == [{**y20, "ended_members": 1, "last_close_in_year": 0}] and cut["ended"] == 1  # C only: the others are priced on row 3


def test_data_report_membership_checks(tmp_path):
    fields, member = _qa_market()
    report = _report(tmp_path, fields, member, qa={"membership_checks": True})
    dates = fields["close"].index
    checked = dates[(dates >= "2020-07-01") & (dates < "2021-04-01")]
    assert f"Dates checked: the {len(checked)} model dates before the test period, 2020-07-01 .. 2021-03-31; no later date is read." in report
    assert "stocks that are members on a model date before the test period: 9" in report and "stocks ever in the index" not in report
    section = _section(report, "## Membership, pricing and survivorship (docs/RULES.md, Rule ND)")
    # brute force, one stock-day at a time
    m, c = member.loc[checked], fields["close"].loc[checked]
    for year in (2020, 2021):
        rows = [d for d in checked if d.year == year]
        per_day = [int(m.loc[d].sum()) for d in rows]
        days = sum(per_day)
        priced = sum(int((m.loc[d] & c.loc[d].notna()).sum()) for d in rows)
        line = next(x for x in section.splitlines() if x.startswith(f"| {year} |"))
        cells = [x.strip() for x in line.strip("|").split("|")]
        assert cells[:8] == [str(year), str(len(rows)), str(min(per_day)), f"{np.median(per_day):g}", str(max(per_day)), str(days), str(priced), f"{priced / days:.2%}"]
        assert cells[8:] == (["1", "0"] if year == 2020 else ["1", "1"])  # S05's data ends on 2021-02-23 (row 299)
    assert "members per day min 8, median 9, max 9" in section  # S07 never, S08 from row 200, S05 leaves on row 310
    total, unpriced = int(m.to_numpy().sum()), int((m & c.isna()).to_numpy().sum())
    assert f"{total} member-days, {total - unpriced} priced ({(total - unpriced) / total:.2%}), {unpriced} unpriced." in section
    assert unpriced == 3 + 1 + 40 + 10  # S01, S03, the S06 hole, and S05's 10 member-days after its last close
    assert "Survivorship: 9 stocks are members on at least one date checked; the data of 1 of them ends early (0 of those 1 with no close on any date checked)." in section
    # the longest spans, in order: the S06 hole, S05 after its delisting, the S01 suspension, the S03 day
    spans = [x for x in section.split("longest:")[1].splitlines() if x.startswith("| S0")]
    assert spans == [
        f"| S06 | {dates[250].date()} | {dates[289].date()} | 40 | - |",
        f"| S05 | {dates[300].date()} | {dates[309].date()} | 10 | - |",
        f"| S01 | {dates[150].date()} | {dates[152].date()} | 3 | - |",
        f"| S03 | {dates[200].date()} | {dates[200].date()} | 1 | - |",
    ]
    assert "| 1-5 | 2 | 4 |" in section and "| 6-20 | 1 | 10 |" in section and "| 21-60 | 1 | 40 |" in section and "| 251 or more | 0 | 0 |" in section
    # warm-up: the 130 dates loaded before 2020-07-01; the first market state reads the last 120 of them
    assert f"Warm-up (the 130 dates loaded before the first model date, 2020-01-01 .. {dates[129].date()}): members per day min 8, median 8, max 8." in section
    assert f"reads the member returns of the 120 dates {dates[10].date()} .. {dates[129].date()} (the longest window is 120 days)" in section
    assert "Members per day on those dates: min 8, median 8, max 8; members with a one-day return: min 8, median 8." in section
    # no spot prices (they are dated in the test period), and label counts stop 20 days before it
    assert "Left out: this check compares three raw closes dated 2024-03-01" in report and "| Stock | Date | Ours | Expected |" not in report
    assert f"Label counts stop at {dates[325 - 20].date()}" in report and dates[326] == pd.Timestamp("2021-04-01")
    assert "Membership filters" not in section


def test_data_report_with_membership_checks_reads_nothing_from_the_test_period(tmp_path):
    """Rule ND: prices, suspensions and membership from the first test date on must not move a single character."""
    fields, member = _qa_market()
    base = _report(tmp_path / "a", fields, member, qa={"membership_checks": True})
    changed, later = {k: v.copy() for k, v in _shock(fields, slice(326, None)).items()}, member.copy()
    assert fields["close"].index[326] == pd.Timestamp("2021-04-01")  # the first test date
    _blank(changed, slice(326, 400), 2)  # S02 stops trading as the test period starts
    _blank(changed, slice(340, None), 6)  # S06's data ends inside it
    later.iloc[326:, 7] = True  # S07 joins the list in the test period
    later.iloc[330:, 0] = False  # S00 leaves it
    assert _report(tmp_path / "b", changed, later, qa={"membership_checks": True}) == base
    # the same changes one day earlier (the last date checked) do show
    earlier = {k: v.copy() for k, v in fields.items()}
    _blank(earlier, slice(325, 326), 2)
    assert _report(tmp_path / "c", earlier, member, qa={"membership_checks": True}) != base
    # without the flag (the CSI300 report) the label counts do reach into the test period: that report is unchanged
    plain_a = _report(tmp_path / "d", fields, member)
    plain_b = _report(tmp_path / "e", changed, later)
    assert plain_a != plain_b and "Membership, pricing and survivorship" not in plain_a


def test_data_report_without_the_flag_is_the_csi300_layout(tmp_path):
    fields, member = _qa_market()
    report = _report(tmp_path, fields, member)
    assert "stocks ever in the index: 10" in report and "## Spot prices (raw close = $close / $factor)" in report
    assert "Rule ND" not in report and "Label counts stop" not in report
    # a market without paper numbers: no paper column instead of "None%"
    assert "| Regime (agent) | Our coverage of training days |\n|---|---|\n| 1 Low-Vol Bull |" in report
    assert "None" not in report and "Paper (Table 4)" not in report


def test_data_report_shows_what_the_membership_filters_removed(tmp_path):
    fields, member = _qa_market()
    _blank(fields, slice(0, 280), 9)  # S09 lists on row 280, a model date
    member.iloc[:280, 9] = False
    report = _report(tmp_path, fields, member, qa={"membership_checks": True}, data={"min_traded_days": 5, "exclude_codes": ("S00",)})
    section = _section(report, "## Membership, pricing and survivorship (docs/RULES.md, Rule ND)")
    dates = fields["close"].index
    checked = dates[(dates >= "2020-07-01") & (dates < "2021-04-01")]
    listed = int(member.loc[checked].to_numpy().sum())
    s00, s09 = len(checked), 5  # S00 on every date checked; S09's first 5 traded days (every other stock is seasoned in the warm-up)
    assert "Member = in the `all` list on that day, after the membership filters below." in section
    assert f"- Membership filters: the `all` list has {listed} member-days on the dates checked; {listed - s00 - s09} are kept." in section
    assert f"  - Never members (not stocks): S00; {s00} member-days removed." in section
    assert f"once it has 5 traded days in the loaded data up to the day before (traded day 6 on); {s09} member-days removed ({s09 / (listed - s00):.2%} of" in section
    assert "stocks that are members on a model date before the test period: 8" in report  # 10 minus S00 and S07
    assert "Warm-up (the 130 dates loaded before the first model date" in section and "members per day min 0, median 6, max 6." in section  # S01-S06


def test_a_new_listing_filter_longer_than_the_warm_up_fails_loudly(tmp_path):
    """Every stock needs its traded days in the loaded data first: with too little history loaded before the first
    model date the market state has nothing to read, and the build refuses instead of filling the gap."""
    fields, member = _market()  # 130 dates before the first model date
    with pytest.raises(ValueError, match="load more warm-up history"):
        _prepare(tmp_path, fields, member, data={"min_traded_days": 120})
