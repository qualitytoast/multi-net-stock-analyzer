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


def test_price_limits_by_board_and_date():
    codes = ["SH600000", "SZ300750", "SH688981", "SZ000001", "SZ301000"]
    lim = price_limit(codes, pd.DatetimeIndex(["2020-08-21", "2020-08-24"]))
    assert lim[0].tolist() == [0.10, 0.10, 0.20, 0.10, 0.10]  # ChiNext still 10% before 2020-08-24
    assert lim[1].tolist() == [0.10, 0.20, 0.20, 0.10, 0.20]


def test_trade_flags_detect_limit_closes_and_suspensions():
    dates = pd.DatetimeIndex(["2021-03-01", "2021-03-02"])
    cols = ["SH600000", "SH600001", "SH600002", "SH600003", "SZ300001"]
    close = pd.DataFrame([[10.0, 10.0, 10.0, 10.0, 10.0], [11.0, 9.0, 10.96, np.nan, 11.0]], index=dates, columns=cols)
    high = pd.DataFrame([[10.0] * 5, [11.0, 9.5, 11.0, np.nan, 11.2]], index=dates, columns=cols)
    low = pd.DataFrame([[10.0] * 5, [10.5, 9.0, 10.5, np.nan, 10.6]], index=dates, columns=cols)
    f = trade_flags(close, high, low)
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
    f = trade_flags(close, close, close)
    g = code_space_growth(f, dates[:3])
    assert g[:, 0].tolist() == pytest.approx([1.0, 1.0, 1.2])  # the suspended stock earns its move when it trades again
    assert g[:, 1].tolist() == pytest.approx([1.1, 1.0, 1.0])
    pg = panel_growth(np.array([[0, -1], [1, 0]]), np.array([[0.1, np.nan], [np.nan, -0.5]]), 2)
    assert np.allclose(pg, [[1.1, 1.0], [0.5, 1.0]])


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
