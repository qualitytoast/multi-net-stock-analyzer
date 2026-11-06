"""Plan tests 22-27: metrics, backtest timing and size, Shapley interaction."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from detmaso.eval.backtest import top_decile_backtest
from detmaso.eval.metrics import ic_summary, portfolio_metrics
from detmaso.eval.shapley import coalition_values, interaction_index, shapley_weight


# 22 ------------------------------------------------------------------------------------------
def test_metrics_on_toy_series():
    r = np.array([0.1, -0.1])
    m = portfolio_metrics(r, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-03"))
    assert m["mdd"] == pytest.approx(1 - 0.99 / 1.1)
    day1_loss = portfolio_metrics(np.array([-0.1, 0.05]), pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-03"))
    assert day1_loss["mdd"] == pytest.approx(0.1)  # starting value 1 counts as a peak
    assert day1_loss["qlib_mdd"] == pytest.approx(0.0)  # Qlib's convention misses a day-1 loss
    # Compounded annual growth over exactly 4 calendar years (1461 days).
    four_years = np.full(4, 0.1)
    g = portfolio_metrics(four_years, pd.Timestamp("2020-01-01"), pd.Timestamp("2024-01-01"))
    assert g["ar"] == pytest.approx(0.1)
    assert g["days_per_year"] == pytest.approx(1.0)
    assert g["qlib_ar"] == pytest.approx(0.1 * 238)


def test_sharpe_uses_observed_days_per_year():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, size=484)
    m = portfolio_metrics(r, pd.Timestamp("2024-03-01"), pd.Timestamp("2026-03-02"))
    assert m["days_per_year"] == pytest.approx(484 / (731 / 365.25))
    assert m["sharpe"] == pytest.approx(r.mean() / r.std(ddof=1) * math.sqrt(m["days_per_year"]))


def test_ic_summary():
    s = ic_summary(np.array([0.1, 0.0, 0.2, np.nan]))
    assert s["ic"] == pytest.approx(0.1) and s["icir"] == pytest.approx(0.1 / 0.1) and s["ic_days"] == 3


# 23 / 24 -------------------------------------------------------------------------------------
def test_backtest_timing_size_and_missing_returns():
    alpha = np.array([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0]])
    tradable = np.ones_like(alpha, dtype=bool)
    tradable[0, 0] = False  # best score but no price on t: not eligible
    r_next = np.arange(11, dtype=float)[None] / 100
    r_next[0, 2] = np.nan  # suspended on t+1: earns 0
    port, bench, picks = top_decile_backtest(alpha, tradable, r_next, 0.10)
    assert len(picks[0]) == math.ceil(0.1 * 10) == 1
    assert picks[0].tolist() == [1]
    assert port[0] == pytest.approx(0.01)
    port2, _, picks2 = top_decile_backtest(alpha, tradable, r_next, 0.25)
    assert picks2[0].tolist() == [1, 2, 3]
    assert port2[0] == pytest.approx((0.01 + 0.0 + 0.03) / 3)
    assert bench[0] == pytest.approx(np.nan_to_num(r_next[0, 1:]).mean())


# 25-27 ---------------------------------------------------------------------------------------
def _game(n: int, fn) -> np.ndarray:
    v = np.zeros(2**n)
    for mask in range(2**n):
        v[mask] = fn({i for i in range(n) if mask >> i & 1})
    return v


def test_additive_game_has_no_interactions():
    c = np.array([0.3, -0.1, 0.5, 0.2, 0.05])
    I = interaction_index(_game(5, lambda S: sum(c[i] for i in S)), 5)
    np.testing.assert_allclose(I[~np.eye(5, dtype=bool)], 0.0, atol=1e-12)


def test_pure_pair_synergy():
    I = interaction_index(_game(5, lambda S: float({1, 3} <= S)), 5)
    assert I[1, 3] == pytest.approx(1.0)
    others = [(i, j) for i in range(5) for j in range(5) if i != j and {i, j} != {1, 3}]
    assert all(abs(I[i, j]) < 1e-12 for i, j in others)


def test_shapley_symmetry_and_weights():
    rng = np.random.default_rng(0)
    v = rng.normal(size=32)
    v[0] = 0.0
    I = interaction_index(v, 5)
    np.testing.assert_allclose(I, I.T, equal_nan=True)
    total = sum(math.comb(3, s) * shapley_weight(s, 5) for s in range(4))
    assert total == pytest.approx(1.0)
    assert [shapley_weight(s, 5) for s in range(4)] == pytest.approx([1 / 4, 1 / 12, 1 / 12, 1 / 4])


# E7: timing through the production label and date functions ----------------------------------
def test_backtest_return_is_next_days_close_to_close():
    from detmaso.data.labels import next_day_return
    from detmaso.eval.report import next_dates

    dates = pd.bdate_range("2021-01-04", periods=6)
    close = pd.DataFrame({"A": [10, 11, 12, 13, 14, 15.0], "B": [10, 9, 8, 7, 6, 5.0], "C": [10, 10, 10, 20, 20, 20.0]}, index=dates)
    r_next = next_day_return(close).to_numpy()
    t = 2  # score C highest on day 2: its return from close(2) to close(3) is +100%
    alpha = np.array([[0.0, -1.0, 1.0]])
    port, _, picks = top_decile_backtest(alpha, np.ones((1, 3), dtype=bool), r_next[t : t + 1], 0.10)
    assert picks[0].tolist() == [2] and port[0] == pytest.approx(close.iloc[3, 2] / close.iloc[2, 2] - 1)
    assert next_dates(dates[2:3], dates)[0] == dates[3]


# E8: coalition values on a tiny case ----------------------------------------------------------
def test_coalition_values_match_direct_ics():
    from detmaso.losses.stats import daily_spearman

    rng = np.random.default_rng(0)
    T, n, N = 6, 3, 25
    alphas = rng.normal(size=(T, n, N))
    logits = rng.normal(size=(T, n))
    labels = rng.normal(size=(T, N))
    ok = np.ones((T, N), dtype=bool)
    v, degenerate = coalition_values(alphas, logits, labels, ok)
    assert v[0] == 0.0 and np.isfinite(v).all() and degenerate.sum() == 0
    for i in range(n):
        assert v[1 << i] == pytest.approx(daily_spearman(alphas[:, i], labels, ok).mean())
    w = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    full = np.einsum("tn,tnk->tk", w, alphas)
    assert v[(1 << n) - 1] == pytest.approx(daily_spearman(full, labels, ok).mean())
    alphas[:, 1] = 0.0  # a dead agent: its singleton coalition has IC 0 on every day
    v2, degenerate2 = coalition_values(alphas, logits, labels, ok)
    assert v2[1 << 1] == 0.0 and degenerate2[1 << 1] == T


# E9: Qlib oracle and verdict rules ------------------------------------------------------------
def test_qlib_forms_match_qlib_risk_analysis():
    from qlib.contrib.evaluate import risk_analysis

    r = np.random.default_rng(1).normal(0.0005, 0.012, size=300)
    ours = portfolio_metrics(r, pd.Timestamp("2024-01-01"), pd.Timestamp("2025-03-01"))
    ra = risk_analysis(pd.Series(r), N=238, mode="sum")["risk"]
    assert ours["qlib_ar"] == pytest.approx(ra["annualized_return"])
    assert ours["qlib_ir"] == pytest.approx(ra["information_ratio"])
    assert ours["qlib_mdd"] == pytest.approx(-ra["max_drawdown"])


def test_verdict_rules_in_order():
    from detmaso.eval.report import band_status, verdict

    good_q = {f"q{i}_ok": True for i in range(1, 5)}
    base = {"ic": 0.081, "sharpe": 1.85, "ar": 0.315, "mdd": 0.085}
    assert verdict("csi300", base, good_q) == ("Reproduced", [])
    assert verdict("csi300", {**base, "ic": 0.054}, good_q)[0] == "Not reproduced"  # floor first
    assert verdict("sp500", {**base, "ic": 0.050, "sharpe": 1.0}, good_q)[0] != "Not reproduced"  # per-market floor
    partial = verdict("csi300", {**base, "ic": 0.060}, {**good_q, "q2_ok": False})
    assert partial[0] == "Partially reproduced" and "q2" in partial[1]
    assert verdict("csi300", {**base, "ic": 0.060}, {**good_q, "q2_ok": False, "q3_ok": False})[0] == "Inconclusive"
    assert band_status("ic", 0.081 + 0.016, 0.081) == "exceeds paper"
    assert band_status("ic", 0.081 - 0.015, 0.081) == "reproduced band"
    assert band_status("mdd", 0.085 + 0.05, 0.085) == "close band"
    assert band_status("mdd", 0.02, 0.085) == "exceeds paper"
