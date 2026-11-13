"""Trading the scores: turnover, costs, blocked trades and the long-short spread (docs/RULES.md, Rule WT).

The paper backtest (eval.backtest.top_decile_backtest) is left untouched; this module only adds what it leaves out.

- Turnover: each day's buys and sells, as shares of the portfolio, against the previous holdings after they
  drifted with the day's returns. One-way turnover = the share of the portfolio replaced.
- Costs: buy share x buy rate + sell share x sell rate, paid at the rebalance; net = (1 - cost)(1 + gross) - 1.
- Without blocking (``simulate_unblocked``): exactly the paper backtest's portfolio and gross returns (equal
  weights on the day's picks; a pick with no next-day price earns 0 that day), plus its turnover.
- With blocking (``simulate_blocked``): at the close of day t a stock that did not trade on t, or closed at its
  lower price limit, cannot be sold or reduced, and a stock that closed at its upper limit cannot be bought or
  increased. Weight that cannot be placed stays in cash (return 0); the other picks share the rest equally.
  The limits are the market's own, from the config (``trading.price_limits``; the default is China's).
  Holdings are valued at their last traded close, so a suspended stock earns its whole move when it trades again.
- Long-short spread: top 10% minus bottom 10% of the eligible stocks, equal weights, gross, never blocked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from detmaso.config import TradingConfig
from detmaso.data.returns import prev_traded_close

LIMIT_TOL = 0.005  # "at the limit": within 0.5 percentage points of it
_PRICE_TOL = 1e-6  # relative tolerance for "the close equals the day's high / low" on adjusted prices


# Price limits ---------------------------------------------------------------------------------------------
def price_limit(codes: list[str], dates: pd.DatetimeIndex, tc: TradingConfig) -> np.ndarray:
    """[T, N] daily price limit as a fraction, from the config's steps (``trading.price_limits``).

    Steps are applied in order: each sets its limit for the codes that start with one of its prefixes (every
    code if it has none), from its first date on, overriding earlier steps. A (code, date) that no step covers
    is an error. China's default: 10%; 20% for STAR (SH688); 20% for ChiNext (SZ300, SZ301) from 2020-08-24.
    """
    dates = pd.DatetimeIndex(dates)
    upper = [str(c).upper() for c in codes]
    lim = np.full((len(dates), len(codes)), np.nan)
    for prefixes, first, limit in tc.price_limits:
        cols = np.array([not prefixes or c.startswith(prefixes) for c in upper], dtype=bool)
        lim[np.ix_(np.asarray(dates >= pd.Timestamp(first)), cols)] = limit
    missing = np.isnan(lim)
    if missing.any():
        t, j = np.argwhere(missing)[0]
        raise ValueError(
            f"no price limit is defined for {int(missing.sum())} (stock, date) pairs, e.g. {codes[j]} on {dates[t].date()}: "
            "add a step to trading.price_limits that covers them"
        )
    return lim


@dataclass
class TradeFlags:
    """Per (date, stock code) on the raw calendar: can it be bought / sold at that day's close?"""

    dates: pd.DatetimeIndex
    codes: list[str]
    traded: np.ndarray  # [T, N] bool
    limit_up: np.ndarray  # [T, N] bool: closed at the upper price limit
    limit_down: np.ndarray  # [T, N] bool: closed at the lower price limit
    last_close: np.ndarray  # [T, N] last traded close (adjusted), NaN before the first trade

    @property
    def can_buy(self) -> np.ndarray:
        return self.traded & ~self.limit_up

    @property
    def can_sell(self) -> np.ndarray:
        return self.traded & ~self.limit_down


def trade_flags(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, tc: TradingConfig) -> TradeFlags:
    """Limit and suspension flags from adjusted daily prices (wide frames, NaN = did not trade)."""
    ret = (close / prev_traded_close(close) - 1.0).to_numpy(dtype=np.float64)
    c, h, lo = (f.to_numpy(dtype=np.float64) for f in (close, high, low))
    lim = price_limit(list(close.columns), close.index, tc)
    with np.errstate(invalid="ignore"):
        up = (ret >= lim - LIMIT_TOL) & (c >= h * (1 - _PRICE_TOL))
        down = (ret <= -(lim - LIMIT_TOL)) & (c <= lo * (1 + _PRICE_TOL))
    return TradeFlags(close.index, list(close.columns), ~np.isnan(c), up, down, close.ffill().to_numpy(dtype=np.float64))


# Picking -------------------------------------------------------------------------------------------------
def ranked_eligible(alpha_hat: np.ndarray, tradable: np.ndarray) -> np.ndarray:
    """Eligible panel columns (tradable, finite score) in descending score order (the paper backtest's order)."""
    elig = np.flatnonzero(tradable & np.isfinite(alpha_hat))
    return elig[np.argsort(-alpha_hat[elig], kind="stable")]


def long_short(alpha_hat: np.ndarray, tradable: np.ndarray, r_next: np.ndarray, top_frac: float = 0.10) -> np.ndarray:
    """[T] gross daily return of top ``top_frac`` minus bottom ``top_frac`` (equal weights; no price = 0)."""
    out = np.full(alpha_hat.shape[0], np.nan)
    for t in range(alpha_hat.shape[0]):
        order = ranked_eligible(alpha_hat[t], tradable[t])
        if len(order) == 0:
            continue
        k = math.ceil(top_frac * len(order))
        r = np.nan_to_num(r_next[t], nan=0.0)
        out[t] = r[order[:k]].mean() - r[order[-k:]].mean()
    return out


# Portfolios ----------------------------------------------------------------------------------------------
@dataclass
class Execution:
    """Per decision date t: the gross return from close t to close t+1 and the trades made at close t."""

    port: np.ndarray  # gross return
    buy: np.ndarray  # share of the portfolio bought
    sell: np.ndarray  # share of the portfolio sold
    cash: np.ndarray  # share left in cash after the rebalance
    blocked_buys: np.ndarray  # picks that could not be raised to their weight
    blocked_sells: np.ndarray  # holdings that could not be sold down

    @property
    def turnover(self) -> np.ndarray:
        """One-way turnover: the share of the portfolio replaced (the larger of buys and sells)."""
        return np.maximum(self.buy, self.sell)


def _trades(h: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    d = w - h
    return float(d[d > 0].sum()), float(-d[d < 0].sum())


def simulate_unblocked(picks: list[np.ndarray], pick_returns: list[np.ndarray], n_codes: int) -> Execution:
    """The paper backtest's portfolio (equal weights on each day's picks, no price = 0) with its turnover.

    picks[t]: stock codes (indices into the code list) picked at close t; pick_returns[t]: their next-day returns.
    """
    T = len(picks)
    port, buy, sell = np.full(T, np.nan), np.zeros(T), np.zeros(T)
    h = np.zeros(n_codes)
    for t in range(T):
        p = np.asarray(picks[t], dtype=np.int64)
        w = np.zeros(n_codes)
        if len(p) == 0:  # no stock to hold (never happens on real data): all cash
            buy[t], sell[t] = _trades(h, w)
            h = w
            continue
        w[p] = 1.0 / len(p)
        buy[t], sell[t] = _trades(h, w)
        r_in = np.nan_to_num(np.asarray(pick_returns[t]), nan=0.0)
        port[t] = r_in.mean()  # in the input's precision, exactly as the paper backtest averages
        r = r_in.astype(np.float64)
        h = np.zeros(n_codes)
        h[p] = (1.0 + r) / len(p) / (1.0 + port[t])
    zeros = np.zeros(T)
    return Execution(port, buy, sell, zeros.copy(), zeros.copy(), zeros.copy())


def rebalance_blocked(h: np.ndarray, picks: np.ndarray, can_buy: np.ndarray, can_sell: np.ndarray) -> tuple[np.ndarray, int, int]:
    """New weights from drifted holdings ``h`` toward equal weights on ``picks``, respecting blocked trades.

    A weight that would have to fall but cannot be sold, or rise but cannot be bought, stays at its current
    value; the remaining picks share what is left equally (repeated until no pick is blocked). Returns
    (weights, picks or holdings blocked from rising, holdings blocked from falling).
    """
    w = np.zeros_like(h)
    target = set(int(j) for j in picks)
    pinned: dict[int, float] = {}
    blocked_buys = blocked_sells = 0
    for j in np.flatnonzero(h > 0):
        if int(j) not in target and not can_sell[j]:
            pinned[int(j)] = h[j]
            blocked_sells += 1
    free = [int(j) for j in picks]
    q = 0.0
    while free:
        q = (1.0 - sum(pinned.values())) / len(free)
        stuck = [j for j in free if (q > h[j] and not can_buy[j]) or (q < h[j] and not can_sell[j])]
        if not stuck:
            break
        for j in stuck:
            pinned[j] = h[j]
            if q > h[j]:
                blocked_buys += 1
            else:
                blocked_sells += 1
        free = [j for j in free if j not in pinned]
    for j in free:
        w[j] = q
    for j, v in pinned.items():
        w[j] = v
    return w, blocked_buys, blocked_sells


def simulate_blocked(picks: list[np.ndarray], growth: np.ndarray, can_buy: np.ndarray, can_sell: np.ndarray) -> Execution:
    """The top-decile portfolio when blocked trades are respected.

    growth[t, j]: value multiple of stock j from close t to close t+1 (last traded closes; 1 if unknown);
    can_buy / can_sell[t, j]: whether j can be bought / sold at close t. All arrays are in code space.
    """
    T, n_codes = growth.shape
    port, buy, sell, cash = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
    bb, bs = np.zeros(T, dtype=int), np.zeros(T, dtype=int)
    h = np.zeros(n_codes)
    for t in range(T):
        w, bb[t], bs[t] = rebalance_blocked(h, np.asarray(picks[t], dtype=np.int64), can_buy[t], can_sell[t])
        buy[t], sell[t] = _trades(h, w)
        cash[t] = max(0.0, 1.0 - w.sum())
        value = float((w * growth[t]).sum() + cash[t])
        port[t] = value - 1.0
        h = w * growth[t] / value
    return Execution(port, buy, sell, cash, bb, bs)


def code_space_growth(flags: TradeFlags, decision_dates: pd.DatetimeIndex) -> np.ndarray:
    """[T, N] value multiple from close t to the next trading day's close, on last traded closes (1 if unknown)."""
    pos = flags.dates.get_indexer(decision_dates)
    if (pos < 0).any() or (pos + 1 >= len(flags.dates)).any():
        raise KeyError("decision dates outside the loaded calendar")
    c0, c1 = flags.last_close[pos], flags.last_close[pos + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        g = c1 / c0
    return np.where(np.isfinite(g), g, 1.0)


def panel_growth(code_idx: np.ndarray, r_next: np.ndarray, member: np.ndarray, n_codes: int) -> np.ndarray:
    """[T, N codes] value multiples from a placebo panel's next-day returns: 1 for a stock outside the day's
    universe, outside that day's shuffle (``member`` False: the shuffled label_ok, see data.placebo) or with no
    price. Only members carry a shuffled next-day return, so no real return may enter."""
    g = np.ones((code_idx.shape[0], n_codes))
    for t in range(code_idx.shape[0]):
        shuffled = member[t] & (code_idx[t] >= 0)
        g[t, code_idx[t, shuffled]] = 1.0 + np.nan_to_num(r_next[t, shuffled], nan=0.0)
    return g


# Costs ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CostSetting:
    name: str
    flat_bps: float | None  # None = the realistic, date-aware setting


def cost_settings(tc: TradingConfig) -> list[CostSetting]:
    return [CostSetting(f"{b:g} bps", b) for b in tc.flat_bps] + [CostSetting("realistic", None)]


def stamp_duty_bps(tc: TradingConfig, dates: pd.DatetimeIndex) -> np.ndarray:
    steps = [(pd.Timestamp(d), b) for d, b in tc.stamp_duty_bps]
    if len(dates) and dates.min() < steps[0][0]:
        raise ValueError(f"no stamp duty rate is defined before {steps[0][0].date()} (asked for {dates.min().date()})")
    out = np.zeros(len(dates))
    for start, bps in steps:
        out[np.asarray(dates >= start)] = bps
    return out


def cost_rates(tc: TradingConfig, setting: CostSetting, dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """(buy rate, sell rate) per trade date, as fractions of the traded value."""
    if setting.flat_bps is not None:
        r = np.full(len(dates), setting.flat_bps / 1e4)
        return r, r.copy()
    side = (tc.fee_bps + tc.slippage_bps) / 1e4
    return np.full(len(dates), side), side + stamp_duty_bps(tc, dates) / 1e4


def net_returns(gross: np.ndarray, buy: np.ndarray, sell: np.ndarray, buy_rate: np.ndarray, sell_rate: np.ndarray) -> np.ndarray:
    """Net daily return: the day's trading cost is paid at the rebalance, before the holding period."""
    cost = buy * buy_rate + sell * sell_rate
    return (1.0 - cost) * (1.0 + gross) - 1.0
