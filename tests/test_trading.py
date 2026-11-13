"""Trading tools (docs/RULES.md, Rule WT): price limits, turnover, blocked trades, costs and the long-short spread."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from detmaso.config import TradingConfig
from detmaso.eval.backtest import top_decile_backtest
from detmaso.eval.trading import (
    CostSetting, code_space_growth, cost_rates, cost_settings, long_short, net_returns, panel_growth, price_limit,
    rebalance_blocked, simulate_blocked, simulate_unblocked, trade_flags,
)  # fmt: skip


CHINA = TradingConfig()  # the default price limits are China's


def test_price_limits_by_board_and_date():
    codes = ["SH600000", "SZ300750", "SH688981", "SZ000001", "SZ301000"]
    lim = price_limit(codes, pd.DatetimeIndex(["2020-08-21", "2020-08-24"]), CHINA)
    assert lim[0].tolist() == [0.10, 0.10, 0.20, 0.10, 0.10]  # ChiNext still 10% before 2020-08-24
    assert lim[1].tolist() == [0.10, 0.20, 0.20, 0.10, 0.20]


def _price_limit_before_the_config(codes: list[str], dates: pd.DatetimeIndex) -> np.ndarray:
    """The hard-coded China rule that eval.trading.price_limit used before limits moved into the config."""
    lim = np.full((len(dates), len(codes)), 0.10)
    star = np.array([c.upper().startswith("SH688") for c in codes])
    chinext = np.array([c.upper().startswith(("SZ300", "SZ301")) for c in codes])
    lim[:, star] = 0.20
    lim[np.ix_(np.asarray(dates >= pd.Timestamp("2020-08-24")), chinext)] = 0.20
    return lim


def test_default_price_limits_reproduce_the_hard_coded_china_rule():
    assert CHINA.price_limits == (((), "1990-01-01", 0.10), (("SH688",), "1990-01-01", 0.20), (("SZ300", "SZ301"), "2020-08-24", 0.20))
    codes = [
        "SH600000", "SH601318", "SH603288", "SH605117", "SH688001", "SH688981", "SH689009", "SH000300", "SH900901",
        "SZ000001", "SZ001979", "SZ002594", "SZ003816", "SZ300001", "SZ300750", "SZ301000", "SZ301269", "SZ302132", "SZ399300",
        "BJ430017", "BJ830799", "BJ920002", "sz300750", "sh688981", "sz000001", "S00", "005930",
    ]  # fmt: skip
    dates = pd.DatetimeIndex(["1990-12-19", "2000-01-04", "2019-07-22", "2020-08-21", "2020-08-23", "2020-08-24", "2020-08-25", "2026-04-30"])
    dates = dates.union(pd.bdate_range("2007-01-01", "2026-04-30", freq="29B"))
    new, old = price_limit(codes, dates, CHINA), _price_limit_before_the_config(codes, dates)
    assert new.shape == (len(dates), len(codes)) and np.array_equal(new, old)
    assert set(np.unique(new)) == {0.10, 0.20}


def test_price_limits_follow_the_config_steps_in_order():
    # Korea: 15% for every stock, then 30% from 2015-06-15
    korea = TradingConfig(price_limits=[[[], "1998-12-07", 0.15], [[], "2015-06-15", 0.30]])
    assert korea.price_limits == (((), "1998-12-07", 0.15), ((), "2015-06-15", 0.30))
    lim = price_limit(["005930", "000660"], pd.DatetimeIndex(["1998-12-07", "2015-06-12", "2015-06-15", "2024-02-29"]), korea)
    assert lim.tolist() == [[0.15, 0.15], [0.15, 0.15], [0.30, 0.30], [0.30, 0.30]]
    # a later step wins for the codes it matches, from its first date on; other codes and earlier dates keep theirs
    china_bj = TradingConfig(price_limits=(*CHINA.price_limits, (("BJ",), "1990-01-01", 0.30)))
    codes = ["SH600000", "BJ430017", "SZ300750", "bj920002"]
    lim = price_limit(codes, pd.DatetimeIndex(["2020-08-21", "2021-11-15"]), china_bj)
    assert lim.tolist() == [[0.10, 0.30, 0.10, 0.30], [0.10, 0.30, 0.20, 0.30]]
    back = TradingConfig(price_limits=[[[], "2000-01-01", 0.10], [["A"], "2010-01-01", 0.20], [[], "2020-01-01", 0.05]])
    lim = price_limit(["A1", "B1"], pd.DatetimeIndex(["2005-01-03", "2015-01-05", "2021-01-04"]), back)
    assert lim.tolist() == [[0.10, 0.10], [0.20, 0.10], [0.05, 0.05]]  # the last step overrides the prefix step too
    # the default drives the flags: the same +20% close is a limit close under 20% and not under 30%
    dates = pd.DatetimeIndex(["2021-03-01", "2021-03-02"])
    close = pd.DataFrame({"SZ300001": [10.0, 12.0]}, index=dates)
    assert trade_flags(close, close, close, CHINA).limit_up[1, 0]
    assert not trade_flags(close, close, close, TradingConfig(price_limits=[[[], "1990-01-01", 0.30]])).limit_up[1, 0]


def test_a_stock_day_without_a_price_limit_is_an_error():
    late = TradingConfig(price_limits=[[[], "2015-06-15", 0.30]])
    with pytest.raises(ValueError, match=r"no price limit is defined for 2 \(stock, date\) pairs, e.g. 005930 on 2015-06-12"):
        price_limit(["005930", "000660"], pd.DatetimeIndex(["2015-06-12", "2015-06-15"]), late)
    only_star = TradingConfig(price_limits=[[["SH688"], "1990-01-01", 0.20]])
    with pytest.raises(ValueError, match="SH600000 on 2021-03-01"):
        price_limit(["SH688981", "SH600000"], pd.DatetimeIndex(["2021-03-01"]), only_star)
    assert price_limit(["SH688981"], pd.DatetimeIndex(["2021-03-01"]), only_star).tolist() == [[0.20]]
    with pytest.raises(ValueError, match="no price limit"):  # China's default starts in 1990
        price_limit(["SH600000"], pd.DatetimeIndex(["1989-12-29"]), CHINA)
    dates = pd.DatetimeIndex(["2015-06-12", "2015-06-15"])
    close = pd.DataFrame({"005930": [10.0, 11.0]}, index=dates)
    with pytest.raises(ValueError, match="no price limit"):
        trade_flags(close, close, close, late)


@pytest.mark.parametrize(
    "steps, message",
    [
        ([], "at least one"),
        ([[[], "1990-01-01"]], "steps are"),
        (["SH688"], "steps are"),
        ([["SH688", "1990-01-01", 0.2]], "must be a list"),
        ([[[""], "1990-01-01", 0.2]], "prefix is empty"),
        ([[[], "1990-01-01", 10]], "fraction between 0 and 1"),
        ([[[], "1990-01-01", 0.0]], "fraction between 0 and 1"),
    ],
)
def test_invalid_price_limit_settings_raise(steps, message):
    with pytest.raises(ValueError, match=message):
        TradingConfig(price_limits=steps)


def test_price_limits_load_from_yaml_and_are_never_hashed(tmp_path):
    from detmaso.config import load_config
    from tests.conftest import ROOT

    paper = load_config(ROOT / "configs/csi300.yaml")
    assert paper.trading.price_limits == CHINA.price_limits and paper.trading.max_price_limit == 0.20
    (tmp_path / "paper.yaml").write_text((ROOT / "configs/csi300.yaml").read_text())
    (tmp_path / "explicit.yaml").write_text(
        'base: paper.yaml\ntrading:\n  price_limits: [[[], "1990-01-01", 0.10], [["SH688"], "1990-01-01", 0.20], [["SZ300", "SZ301"], "2020-08-24", 0.20]]\n'
    )
    (tmp_path / "korea.yaml").write_text('base: paper.yaml\ntrading:\n  price_limits: [[[], 1998-12-07, 0.15], [[], "2015-06-15", 0.30]]\n')
    explicit, korea = load_config(tmp_path / "explicit.yaml"), load_config(tmp_path / "korea.yaml")
    assert explicit == paper  # the YAML form of the default is the default
    assert korea.trading.price_limits == (((), "1998-12-07", 0.15), ((), "2015-06-15", 0.30)) and korea.trading.max_price_limit == 0.30
    assert korea.hash() == paper.hash() and korea.data_hash() == paper.data_hash()  # scoring only: nothing is retrained
    assert korea.trading.fee_bps == paper.trading.fee_bps and korea.trading.stamp_duty_bps == paper.trading.stamp_duty_bps


def test_trade_flags_detect_limit_closes_and_suspensions():
    dates = pd.DatetimeIndex(["2021-03-01", "2021-03-02"])
    cols = ["SH600000", "SH600001", "SH600002", "SH600003", "SZ300001"]
    close = pd.DataFrame([[10.0, 10.0, 10.0, 10.0, 10.0], [11.0, 9.0, 10.96, np.nan, 11.0]], index=dates, columns=cols)
    high = pd.DataFrame([[10.0] * 5, [11.0, 9.5, 11.0, np.nan, 11.2]], index=dates, columns=cols)
    low = pd.DataFrame([[10.0] * 5, [10.5, 9.0, 10.5, np.nan, 10.6]], index=dates, columns=cols)
    f = trade_flags(close, high, low, CHINA)
    assert f.limit_up[1].tolist() == [True, False, False, False, False]  # 10.96 is below the day's high; ChiNext limit is 20%
    assert f.limit_down[1].tolist() == [False, True, False, False, False]
    assert f.traded[1].tolist() == [True, True, True, False, True]
    assert f.can_buy[1].tolist() == [False, True, True, False, True]
    assert f.can_sell[1].tolist() == [True, False, True, False, True]
    assert f.last_close[1, 3] == 10.0  # a suspended stock keeps its last traded close


def test_unblocked_turnover_by_hand():
    ex = simulate_unblocked([np.array([0, 1]), np.array([1, 2])], [np.array([0.1, -0.1]), np.array([0.0, 0.0])], n_codes=3)
    assert ex.port.tolist() == pytest.approx([0.0, 0.0])
    # day 0: buy everything; day 1: drifted (0.55, 0.45, 0) -> (0, 0.5, 0.5)
    assert ex.buy.tolist() == pytest.approx([1.0, 0.55]) and ex.sell.tolist() == pytest.approx([0.0, 0.55])
    assert ex.turnover.tolist() == pytest.approx([1.0, 0.55])


def test_unblocked_portfolio_is_the_paper_backtest():
    rng = np.random.default_rng(0)
    T, N = 40, 30
    alpha, r = rng.normal(size=(T, N)), rng.normal(0, 0.02, size=(T, N))
    r[rng.random((T, N)) < 0.05] = np.nan  # some stocks without a next-day price
    tradable = rng.random((T, N)) > 0.1
    port, _, picks = top_decile_backtest(alpha, tradable, r)
    ex = simulate_unblocked(picks, [r[t, p] for t, p in enumerate(picks)], n_codes=N)
    assert np.allclose(ex.port, port, atol=1e-15)
    assert np.all((ex.turnover >= 0) & (ex.turnover <= 1 + 1e-12))
    assert np.allclose(ex.buy[1:], ex.sell[1:])  # fully invested: every buy is paid for by a sale


def test_rebalance_respects_blocked_trades():
    yes, no = np.ones(3, dtype=bool), np.zeros(3, dtype=bool)
    h = np.array([0.6, 0.4, 0.0])
    w, bb, bs = rebalance_blocked(h, np.array([1, 2]), yes, yes)
    assert w.tolist() == pytest.approx([0.0, 0.5, 0.5]) and (bb, bs) == (0, 0)
    # stock 0 cannot be sold: it stays; the picks share the rest
    w, bb, bs = rebalance_blocked(h, np.array([1, 2]), yes, np.array([False, True, True]))
    assert w.tolist() == pytest.approx([0.6, 0.2, 0.2]) and (bb, bs) == (0, 1)
    # pick 2 is at its upper limit: it cannot be bought, so pick 1 takes everything
    w, bb, bs = rebalance_blocked(h, np.array([1, 2]), np.array([True, True, False]), yes)
    assert w.tolist() == pytest.approx([0.0, 1.0, 0.0]) and (bb, bs) == (1, 0)
    # nothing can be bought and nothing is held: all cash
    w, bb, _ = rebalance_blocked(np.zeros(3), np.array([0, 1]), no, yes)
    assert w.sum() == 0 and bb == 2


def test_blocked_portfolio_by_hand():
    growth = np.array([[1.1, 1.0], [0.9, 1.2], [1.0, 1.0]])
    can = np.ones((3, 2), dtype=bool)
    can_sell = can.copy()
    can_sell[1, 0] = False  # day 1: stock 0 closes at its lower limit and cannot be sold
    ex = simulate_blocked([np.array([0]), np.array([1]), np.array([1])], growth, can, can_sell)
    assert ex.port.tolist() == pytest.approx([0.1, -0.1, 0.0])  # day 1 still holds stock 0
    assert ex.blocked_sells.tolist() == [0, 1, 0]
    assert ex.buy.tolist() == pytest.approx([1.0, 0.0, 1.0]) and ex.sell.tolist() == pytest.approx([0.0, 0.0, 1.0])
    # without blocks, the same picks and growth give the unblocked portfolio
    free = simulate_blocked([np.array([0]), np.array([1]), np.array([1])], growth, can, can)
    unb = simulate_unblocked([np.array([0]), np.array([1]), np.array([1])], [np.array([0.1]), np.array([0.2]), np.array([0.0])], 2)
    assert np.allclose(free.port, unb.port) and np.allclose(free.buy, unb.buy) and np.allclose(free.sell, unb.sell)


def test_growth_uses_last_traded_closes():
    dates = pd.bdate_range("2021-03-01", periods=4)
    close = pd.DataFrame({"A": [10.0, np.nan, np.nan, 12.0], "B": [5.0, 5.5, 5.5, 5.5]}, index=dates)
    f = trade_flags(close, close, close, CHINA)
    g = code_space_growth(f, dates[:3])
    assert g[:, 0].tolist() == pytest.approx([1.0, 1.0, 1.2])  # the suspended stock earns its move when it trades again
    assert g[:, 1].tolist() == pytest.approx([1.1, 1.0, 1.0])


def test_placebo_growth_comes_from_shuffled_returns_only():
    """Rule WT: a placebo holding earns its shuffled next-day return, or 0 when it has none (outside the day's
    universe, outside the shuffle, or no price); a stock outside the shuffle still carries its real return."""
    code_idx = np.array([[0, -1, 2], [1, 0, 2]])  # panel columns -> stock codes; day 1 lists stock 1 before stock 0
    r_next = np.array([[0.1, np.nan, 0.2], [np.nan, -0.5, -0.3]])
    member = np.array([[True, False, True], [True, True, False]])  # stock 2 leaves the shuffle on day 1: its -30% is real
    pg = panel_growth(code_idx, r_next, member, 3)
    assert np.allclose(pg, [[1.1, 1.0, 1.2], [0.5, 1.0, 1.0]])
    # stock 2 is bought on day 0 and cannot be sold on day 1: the whole portfolio is stuck in it, and it earns 0
    can = np.ones((2, 3), dtype=bool)
    can_sell = can.copy()
    can_sell[1, 2] = False
    ex = simulate_blocked([np.array([2]), np.array([0])], pg, can, can_sell)
    assert ex.port.tolist() == pytest.approx([0.2, 0.0]) and ex.blocked_sells.tolist() == [0, 1]


def test_cost_rates_and_the_stamp_duty_switch():
    tc = TradingConfig()
    assert [s.name for s in cost_settings(tc)] == ["0 bps", "10 bps", "20 bps", "realistic"]
    dates = pd.DatetimeIndex(["2023-08-25", "2023-08-28"])
    buy, sell = cost_rates(tc, CostSetting("realistic", None), dates)
    assert buy.tolist() == pytest.approx([0.0008, 0.0008]) and sell.tolist() == pytest.approx([0.0018, 0.0013])
    buy, sell = cost_rates(tc, CostSetting("10 bps", 10.0), dates)
    assert buy.tolist() == sell.tolist() == pytest.approx([0.001, 0.001])
    with pytest.raises(ValueError, match="no stamp duty"):
        cost_rates(tc, CostSetting("realistic", None), pd.DatetimeIndex(["2008-01-02"]))
    net = net_returns(np.array([0.01]), np.array([0.5]), np.array([0.5]), np.array([0.001]), np.array([0.002]))
    assert net[0] == pytest.approx((1 - 0.0015) * 1.01 - 1)


def test_long_short_is_top_minus_bottom_decile():
    alpha = np.arange(10, dtype=float)[None, :]
    r = np.linspace(-0.05, 0.04, 10)[None, :]
    assert long_short(alpha, np.ones((1, 10), dtype=bool), r)[0] == pytest.approx(0.04 - (-0.05))
