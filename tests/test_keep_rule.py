"""The walk-forward keep rule (Rule WK), the placebo rule (Rule WP), the stratified paired resampling behind them,
and `compare --split walkforward` applying them to planted run groups."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from detmaso.config import PlaceboConfig, SplitConfig, WalkForwardConfig
from detmaso.eval.guard import IMPROVEMENT, NO_IMPROVEMENT, WORSE, keep_rule, rule_wp
from detmaso.eval.wf_compare import advantage, run_wf_compare
from detmaso.eval.wf_stats import RESULTS_FILE, SERIES_FILE, WFError, bootstrap, frame_to_run, seed_metrics
from tests.conftest import synthetic_config

YEARS = list(range(2013, 2024))


# The rule itself ---------------------------------------------------------------------------------------------
def _inputs(gains, lows=None, highs=None, avg=(0.01, 0.002, 0.02), seed_gains=(1, 1, 1, 1, 1), sharpe=(0.1, -0.2, 0.4)):
    lows = lows or [g - 0.01 for g in gains]
    highs = highs or [g + 0.01 for g in gains]
    yearly = {y: {"gain": g, "low": lo, "high": hi} for y, g, lo, hi in zip(YEARS, gains, lows, highs)}
    return {
        "yearly": yearly,
        "average": dict(zip(("gain", "low", "high"), avg)),
        "seed_gains": dict(enumerate(seed_gains)),
        "sharpe": dict(zip(("diff", "low", "high"), sharpe)),
    }


def test_all_conditions_give_an_improvement():
    x = _inputs([0.01] * 8 + [-0.01] * 3)
    k = keep_rule(**x, has_setting=False)
    assert k["verdict"] == IMPROVEMENT and k["k2_detail"]["wins_needed"] == 8 and len(k["year_wins"]) == 8


@pytest.mark.parametrize(
    "change, failing",
    [
        ({"gains": [0.01] * 7 + [-0.01] * 4}, "k2"),  # 7 of 11 years
        ({"avg": (0.01, -0.001, 0.02)}, "k1"),  # average range touches 0
        ({"seed_gains": (1, 1, 1, -1, -1)}, "k3"),  # 3 of 5 seeds
        ({"sharpe": (-0.5, -0.9, -0.1)}, "k5"),  # after-cost Sharpe detectably worse
    ],
)
def test_each_condition_can_block_an_improvement(change, failing):
    args = {"gains": [0.01] * 11}
    args.update(change)
    k = keep_rule(**_inputs(**args), has_setting=False)
    assert k["verdict"] == NO_IMPROVEMENT and not k[failing]
    assert all(k[c] for c in ("k1", "k2", "k3", "k4", "k5") if c != failing)


def test_worse_and_the_next_setting():
    assert keep_rule(**_inputs([-0.01] * 11, avg=(-0.01, -0.02, -0.001)), has_setting=False)["verdict"] == WORSE
    x = _inputs([0.01] * 11)
    assert keep_rule(**x, has_setting=True, neighbor_gain=0.004)["verdict"] == IMPROVEMENT
    k = keep_rule(**x, has_setting=True, neighbor_gain=-0.001)
    assert k["verdict"] == NO_IMPROVEMENT and not k["k4"]
    with pytest.raises(ValueError, match="next setting"):
        keep_rule(**x, has_setting=True)


def test_targeted_form():
    gains = [0.0] * 11
    targets = [2015, 2016, 2019, 2020]
    for y in (2015, 2016, 2019):  # 3 of 4 targeted years won; other years flat
        gains[YEARS.index(y)] = 0.02
    x = _inputs(gains)
    k = keep_rule(**x, has_setting=False, target_years=targets)
    assert k["verdict"] == IMPROVEMENT and k["k2_detail"]["target_wins_needed"] == 3
    # one other year detectably worse blocks it
    x["yearly"][2022].update(gain=-0.02, low=-0.03, high=-0.005)
    k = keep_rule(**x, has_setting=False, target_years=targets)
    assert not k["k2"] and k["k2_detail"]["other_years_worse"] == [2022]
    with pytest.raises(ValueError, match="not walk-forward test years"):
        keep_rule(**x, has_setting=False, target_years=[2010])


def test_placebo_rule():
    ok = rule_wp({"a": (-0.01, 0.02), "b": (-0.02, 0.0)}, {"a vs b": NO_IMPROVEMENT, "b vs a": WORSE})
    assert ok["pass"]
    assert not rule_wp({"a": (0.001, 0.02), "b": (-0.02, 0.0)}, {"a vs b": NO_IMPROVEMENT})["p1"]
    assert not rule_wp({"a": (-0.01, 0.02)}, {"a vs b": IMPROVEMENT})["p2"]


# The resampling, on synthetic daily series ----------------------------------------------------------------------
def _cfg(tmp_path, variant: str):
    cfg = synthetic_config(tmp_path, variant=variant)
    return dataclasses.replace(
        cfg,
        splits=SplitConfig(train=("2008-01-01", "2022-12-31"), valid=("2023-01-01", "2024-02-28"), test=("2024-03-01", "2026-03-01")),
        walkforward=WalkForwardConfig(),
    )


def _frame(ic_shift: float = 0.0, seed: int = 0, n_days: int = 60, seed_shift=None) -> pd.DataFrame:
    """A synthetic walk-forward daily series: 11 years x n_days, 5 seeds, daily IC around 0.05 + shift."""
    rng = np.random.default_rng(seed)
    rows = []
    for y in YEARS:
        dates = pd.bdate_range(f"{y}-01-02", periods=n_days + 1)
        for s in range(5):
            shift = ic_shift + (0.0 if seed_shift is None else seed_shift[s])
            n = n_days
            rows.append(pd.DataFrame({
                "seed": s, "year": y, "decision_date": dates[:-1], "ret_date": dates[1:], "in_ic": np.arange(n) < n - 20,
                "ic": 0.05 + shift + 0.1 * rng.standard_normal(n), "port": 0.0005 + 0.01 * rng.standard_normal(n), "bench": 0.0003 + np.zeros(n),
                "ls": 0.001 * rng.standard_normal(n), "buy": np.full(n, 0.3), "sell": np.full(n, 0.3), "port_blk": 0.0005 + 0.01 * rng.standard_normal(n),
                "buy_blk": np.full(n, 0.3), "sell_blk": np.full(n, 0.3), "cash_blk": np.zeros(n), "blocked_buys": np.zeros(n), "blocked_sells": np.zeros(n),
                "entropy": np.zeros(n), "w0": np.ones(n),
            }))  # fmt: skip
    frame = pd.concat(rows, ignore_index=True)
    frame.loc[~frame["in_ic"], "ic"] = np.nan
    return frame


def _run(tmp_path, variant: str, **kw):
    return frame_to_run(_frame(**kw), _cfg(tmp_path, variant), f"{variant}.yaml", variant)


def _judge(cand, ref, n_boot=500, **kw):
    boot = bootstrap([ref, cand], n_boot=n_boot)
    x = advantage(cand, ref, {r.name: seed_metrics(r) for r in (ref, cand)}, boot)
    return x, keep_rule(x["yearly"], x["average"], x["seed_gains"], x["sharpe"], has_setting=False, **kw)


def test_a_clearly_better_model_is_an_improvement_and_noise_is_not(tmp_path):
    ref = _run(tmp_path, "ref", seed=1)
    better = _run(tmp_path, "better", ic_shift=0.05, seed=2)
    same = _run(tmp_path, "same", ic_shift=0.0, seed=3)
    assert _judge(better, ref)[1]["verdict"] == IMPROVEMENT
    x, k = _judge(same, ref)
    assert k["verdict"] != IMPROVEMENT and x["average"]["low"] < 0 < x["average"]["high"]


def test_one_good_seed_is_not_enough(tmp_path):
    ref = _run(tmp_path, "ref", seed=1)
    lucky = _run(tmp_path, "lucky", seed=4, seed_shift=[0.4, -0.02, -0.02, -0.02, -0.02])
    _, k = _judge(lucky, ref)
    assert k["seed_wins"] == 1 and not k["k3"] and k["verdict"] != IMPROVEMENT


def test_resampling_is_reproducible_and_independent_of_the_other_groups(tmp_path):
    ref, a, b = _run(tmp_path, "ref", seed=1), _run(tmp_path, "a", seed=2), _run(tmp_path, "b", seed=3)
    one = bootstrap([ref, a], n_boot=300)
    again = bootstrap([ref, a], n_boot=300)
    three = bootstrap([b, a, ref], n_boot=300)
    for k in ("ic_year", "ic_avg", "sharpe_net", "sharpe_gross"):
        assert np.array_equal(one["a"][k], again["a"][k]) and np.array_equal(one["a"][k], three["a"][k])
        assert np.array_equal(one["ref"][k], three["ref"][k])
    # the yearly IC levels resample around the observed seed-mean
    sm = seed_metrics(a)
    lo, hi = np.percentile(one["a"]["ic_year"], [2.5, 97.5], axis=1)
    assert np.all((lo < sm["ic_year"].mean(axis=0)) & (sm["ic_year"].mean(axis=0) < hi))


def test_the_after_cost_sharpe_difference_uses_the_blocked_series(tmp_path):
    """Rule WT: "after-cost Sharpe" = realistic costs with blocking, so K5 reads the blocked portfolio only."""
    frame = _frame(seed=1)
    ref = frame_to_run(frame, _cfg(tmp_path, "ref"), "ref.yaml", "ref")

    def diff(column: str) -> dict:
        lower = frame.copy()
        lower[column] -= 0.01  # 1% a day worse in that portfolio, everything else identical
        return _judge(frame_to_run(lower, _cfg(tmp_path, "lower"), "lower.yaml", "lower"), ref)[0]["sharpe"]

    assert diff("port")["diff"] == 0  # the paper backtest (no blocking) plays no part
    blocked = diff("port_blk")
    assert blocked["diff"] < 0 and blocked["high"] < 0


# `compare --split walkforward`, on planted run groups --------------------------------------------------------
def _plant(root, variant: str, frame: pd.DataFrame, placebo: bool = False) -> tuple:
    """Save a daily series as a fully evaluated run group; returns compare's (config, config path, tag)."""
    cfg = _cfg(root, variant)
    if placebo:
        cfg = dataclasses.replace(cfg, placebo=PlaceboConfig(shuffle_seed=0))
    run = cfg.run_path / variant
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"hash": cfg.hash()}))
    (run / RESULTS_FILE).write_text("{}")
    frame.to_parquet(run / SERIES_FILE, index=False)
    return cfg, f"{variant}.yaml", variant


def _no_signal(seed: int) -> pd.DataFrame:
    """A series whose IC averages exactly 0 in every year and seed: there is nothing to find."""
    frame = _frame(seed=seed)
    frame["ic"] -= frame.groupby(["seed", "year"])["ic"].transform("mean")
    return frame


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    """Run groups with planted IC gains over `ref`: +0.05 (better), +0.02 (up), -0.05 (down); and placebo groups."""
    root = tmp_path_factory.mktemp("planted")
    groups = {
        "ref": _plant(root, "ref", _frame(seed=1)),
        "better": _plant(root, "better", _frame(ic_shift=0.05, seed=2)),
        "up": _plant(root, "up", _frame(ic_shift=0.02, seed=5)),
        "down": _plant(root, "down", _frame(ic_shift=-0.05, seed=6)),
        "placebo_a": _plant(root, "placebo_a", _no_signal(7), placebo=True),
        "placebo_b": _plant(root, "placebo_b", _no_signal(8), placebo=True),
        "placebo_leak": _plant(root, "placebo_leak", _frame(seed=9), placebo=True),  # IC 0.05 on "shuffled answers"
    }
    return root, groups


def _compare(planted, tags: list[str], reference: str, **kw) -> dict:
    root, groups = planted
    out = root / "cmp"
    run_wf_compare([groups[t] for t in tags], reference, out, n_boot=500, **kw)
    return json.loads((out / "comparison.json").read_text())["rules"]


def test_compare_keeps_a_planted_gain_with_a_gaining_neighbor(planted):
    k = _compare(planted, ["ref", "better", "up"], "ref", rule="keep", candidate="better", neighbor="up")
    assert k["verdict"] == IMPROVEMENT and all(k[c] for c in ("k1", "k2", "k3", "k4", "k5"))
    assert (k["candidate"], k["reference"], k["neighbor"]) == ("better", "ref", "up")
    assert k["average"]["gain"] == pytest.approx(0.05, abs=0.015) and k["neighbor_gain"] == pytest.approx(0.02, abs=0.015)
    assert len(k["year_wins"]) == 11 and k["seed_wins"] == 5


def test_compare_blocks_a_planted_gain_when_the_neighbor_does_not_gain(planted):
    k = _compare(planted, ["ref", "better", "down"], "ref", rule="keep", candidate="better", neighbor="down")
    assert k["verdict"] == NO_IMPROVEMENT and not k["k4"] and all(k[c] for c in ("k1", "k2", "k3", "k5"))
    assert k["neighbor_gain"] == pytest.approx(-0.05, abs=0.015)
    # the same groups with no tunable setting declared: the other group plays no part
    k = _compare(planted, ["ref", "better", "down"], "ref", rule="keep", candidate="better", no_setting=True)
    assert k["verdict"] == IMPROVEMENT and not k["has_setting"] and k["neighbor"] is None


def test_compare_gains_are_candidate_minus_reference(planted):
    k = _compare(planted, ["ref", "better"], "better", rule="keep", candidate="ref", no_setting=True)
    assert k["verdict"] == WORSE and not k["k1"] and not k["k2"] and not k["k3"]
    assert (k["candidate"], k["reference"]) == ("ref", "better") and k["average"]["gain"] == pytest.approx(-0.05, abs=0.015)
    assert k["year_wins"] == [] and k["seed_wins"] == 0


def test_compare_passes_declared_target_years_to_the_rule(planted):
    k = _compare(planted, ["ref", "better"], "ref", rule="keep", candidate="better", no_setting=True, target_years=[2015, 2016, 2019])
    assert k["verdict"] == IMPROVEMENT
    assert k["k2_detail"] == {"form": "targeted", "target_years": [2015, 2016, 2019], "target_wins": [2015, 2016, 2019],
                              "target_wins_needed": 2, "other_years_worse": []}  # fmt: skip


def test_compare_applies_the_placebo_rule(planted):
    p = _compare(planted, ["placebo_a", "placebo_b"], "placebo_a", rule="placebo")
    assert p["rule"] == "placebo" and p["p1"] and p["p2"] and p["pass"]
    assert set(p["keep_verdicts"]) == {"placebo_a vs placebo_b", "placebo_b vs placebo_a"}
    assert all(lo < 0 < hi for lo, hi in p["levels"].values())
    leak = _compare(planted, ["placebo_leak", "placebo_b"], "placebo_b", rule="placebo")
    assert not leak["p1"] and not leak["pass"] and leak["levels"]["placebo_leak"][0] > 0


def test_compare_refuses_wrong_tags_and_misused_rules(planted):
    root, groups = planted
    pair = [groups["ref"], groups["better"]]
    with pytest.raises(WFError, match="reference 'nope' is not one of the compared run groups"):
        run_wf_compare(pair, "nope", root / "cmp", n_boot=50)
    with pytest.raises(WFError, match="needs --candidate"):
        run_wf_compare(pair, "ref", root / "cmp", rule="keep", candidate="ref", no_setting=True, n_boot=50)
    with pytest.raises(WFError, match="exactly two placebo run groups"):
        run_wf_compare(pair, "ref", root / "cmp", rule="placebo", n_boot=50)
    with pytest.raises(WFError, match="placebo and real runs are never compared"):
        run_wf_compare([groups["ref"], groups["placebo_a"]], "ref", root / "cmp", n_boot=50)
